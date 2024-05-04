"""
This script is used to fine-tune different pretrained models.
Some parameters are opened, but some not for the simplicity.
"""

import os
import gc
import json
import copy
import torch
import math
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from fastchat.conversation import get_conv_template
from typing import Dict, Sequence, List
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn.parallel import DataParallel
from utilities.data_tools import train_test_split

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
default_conversation = get_conv_template('phoenix')

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

class InstructDataset(Dataset):
    def __init__(self, data: Sequence, tokenizer: transformers.PreTrainedTokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        sources = self.data[index]
        if isinstance(index, int):
            sources = [sources]
        data_dict = preprocess([e['instruction'] for e in sources], [e['conversations'] for e in sources], self.tokenizer)
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        return data_dict

def preprocess(
        headers: Sequence[str],
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        max_length=1024
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    intermediates = []
    for header, source in zip(headers, sources):
        # header = f"{default_conversation.system_message}"
        conversation, intermediate = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
        intermediates.append(intermediate)

    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)

    # keep only machine responses as targets
    assert len(targets) == len(intermediates)
    for target, inters in zip(targets, intermediates):
        mask = torch.zeros_like(target, dtype=torch.bool)
        for inter in inters:
            tokenized = _tokenize_fn(inter, tokenizer)
            start_idx = tokenized["input_ids"][0].size(0) - 1
            end_idx = tokenized["input_ids"][1].size(0)
            mask[start_idx:end_idx] = True
        target[~mask] = IGNORE_INDEX

    input_ids = input_ids[:max_length]
    targets = targets[:max_length]
    return dict(input_ids=input_ids, labels=targets)


def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = DEFAULT_BOS_TOKEN
    END_SIGNAL = DEFAULT_EOS_TOKEN
    conversation = header
    intermediate = []
    for sentence in source:
        from_str = sentence["role"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'
        # store the string w/o and w/ the response
        value = (from_str + ": " + BEGIN_SIGNAL + sentence["value"] + END_SIGNAL)
        if sentence["role"].lower() == "gpt":
            start = conversation + from_str + ": " + BEGIN_SIGNAL
            end = conversation + value
            intermediate.append([start, end])
        if get_conversation:
            conversation += value
    return conversation, intermediate


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def finetune_lora(model_id: str, 
                  device_map: dict, 
                  rank: int, 
                  target_modules: list, 
                  output_path: str,
                  dataset: Sequence,
                  seed: int,
                  train_dataset_size: int,
                  val_dataset_size: int,
                  train_args: dict):
    """Finetune model with lora

    Args:
        model_id (str)
        device_map (dict)
        rank (int): rank in lora
        target_modules (list): finetuned modules
        output_path (str): path to store the finetuned model
        dataset (Sequence): dataset used to finetune
        seed (int): random seed
        train_dataset_size (int)
        val_dataset_size (int)
        train_args (dict): args used for trainer
    """
    # Quantization type (fp4 or nf4), According to QLoRA paper, for training 4-bit base models (e.g. using LoRA adapters) one should use
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = True

    # model_id = "FreedomIntelligence/phoenix-inst-chat-7b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=use_nested_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device_map) # device_map={"": 0}
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # You can try differnt parameter-effient strategy for model trianing, for more info, please check https://github.com/huggingface/peft
    config = LoraConfig(
        r=rank,
        # empirical value for lora_alpha: 2*r
        lora_alpha=2*rank,
        target_modules=target_modules, # target_modules=["query_key_value"]
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    train_dataset, val_dataset, _ = train_test_split(seed, train_dataset_size, val_dataset_size, dataset)        
        
    train_dataset = InstructDataset(train_dataset, tokenizer)
    val_dataset = InstructDataset(val_dataset, tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # Set training parameters
    training_arguments = transformers.TrainingArguments(**train_args)

    model.train()
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()


    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.save_model(output_path)

    # Empty VRAM
    del model
    del trainer
    gc.collect()
    gc.collect()