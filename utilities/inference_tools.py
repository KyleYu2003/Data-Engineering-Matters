"""
This script is used to generate responses with different finetuned models.
Some parameters are opened, but some not for the simplicity.
"""
import os
import json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from fastchat.conversation import get_conv_template

@torch.no_grad()
def generate(model_id: str,
             output_path: str,
             system_messages: list, 
             query_list: list, 
             temperature: float = 0.5,
             repetition_penalty: int = 1,
             return_answer: bool = False) -> list:
    """This function generates responses using the finetuned model.

    Args:
        model_id (str)
        output_path (str): where the finetuned model is stored
        system_message (str)
        query_list (list): The queries from user
        temperature (float, optional): Defaults to 0.5.
        repetition_penalty (int, optional): Defaults to 1.
        return_answer (bool, optional): Defaults to False.

    Returns:
        list: The responses from the model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map={"": 2})
    model = PeftModel.from_pretrained(model, output_path)
    model = model.merge_and_unload()
    model.config.max_length = 512

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, padding_side="left")
    # tokenizer.pad_token = tokenizer.unk_token

    def conv_format(system_message, query):
        conv = get_conv_template('phoenix')
        # change the default prompt to the a Chinese one.
        conv.set_system_message(system_message)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    query_list = [conv_format(system_message, query) for system_message, query in zip(system_messages, query_list)]
    input_ids = tokenizer(query_list, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    n_input, n_seq = input_ids.shape[0], input_ids.shape[-1]
    output_ids = []
    step = 1
    for index in tqdm(range(0, n_input, step)):
        outputs = model.generate(
            input_ids=input_ids[index: min(n_input, index+step)],
            do_sample=False,
            max_new_tokens=256,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        output_ids += outputs
    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    if return_answer:
        return [response.split("Assistant:")[-1].strip() for query, response in zip(query_list, responses)]
    return responses

