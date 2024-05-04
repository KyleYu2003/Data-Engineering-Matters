import os
from utilities.finetune_tools import *
from datasets import load_dataset

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#%%
# finetune phoenix with original dataset
train_args = {"output_dir":"./checkpoints",
            "num_train_epochs":3,
            "per_device_train_batch_size":20,
            "per_device_eval_batch_size":5,
            "gradient_accumulation_steps":2,
            "optim":'paged_adamw_32bit',
            "evaluation_strategy":'steps',
            "save_strategy":'steps',
            "load_best_model_at_end":True, # save the best model
            "metric_for_best_model":'eval_loss',
            "logging_steps":1,
            "learning_rate":2e-6,
            "weight_decay":0.001,
            "max_steps":-1,
            "warmup_ratio":0.03,
            "group_by_length":True,
            "lr_scheduler_type":"cosine",
            "gradient_checkpointing":True,
            "report_to":"none"}

model_id= "FreedomIntelligence/phoenix-inst-chat-7b"
device_map={"": 0}
rank = 8
target_modules = ["query_key_value"]
current_dir = os.getcwd()
output_path = current_dir + "/saved/phoenix/modified_data1/lora"
dataset = load_dataset("FinGPT/fingpt-fineval")
dataset = dataset['train'].map(lambda sample: {"conversations": [{"role": "human", "value": sample['input']}, {"role": "gpt", "value": sample['output']}]}, batched=False)
seed = 1
train_dataset_size = 800
val_dataset_size = min(len(dataset) - train_dataset_size, train_dataset_size//4)

finetune_lora(model_id, 
              device_map,
              rank,
              target_modules,
              output_path,
              dataset,
              seed,
              train_dataset_size,
              val_dataset_size,
              train_args)

#%%
# finetune phoenix with modified data1
train_args = {"output_dir":"./checkpoints",
            "num_train_epochs":1,
            "per_device_train_batch_size":5,
            "per_device_eval_batch_size":2,
            "gradient_accumulation_steps":2,
            "optim":'paged_adamw_32bit',
            "evaluation_strategy":'steps',
            "save_strategy":'steps',
            "load_best_model_at_end":True, # save the best model
            "metric_for_best_model":'eval_loss',
            "logging_steps":1,
            "learning_rate":2e-6,
            "weight_decay":0.001,
            "max_steps":-1,
            "warmup_ratio":0.03,
            "group_by_length":True,
            "lr_scheduler_type":"cosine",
            "gradient_checkpointing":True,
            "report_to":"none"}

model_id= "FreedomIntelligence/phoenix-inst-chat-7b"
device_map={"": 0}
rank = 8
target_modules = ["query_key_value"]
current_dir = os.getcwd()
output_path = current_dir + "/saved/phoenix/modified_data1/lora"
dataset = json.load(open('data/modified_fineval_with_first_5_items.json'))
dataset = list(map(lambda sample: {"instruction": sample["instruction"], "conversations": [{"role": "human", "value": sample['input']}, {"role": "gpt", "value": sample['output']}]}, dataset))
seed = 1
train_dataset_size = 800
val_dataset_size = min(len(dataset) - train_dataset_size, train_dataset_size//4)

finetune_lora(model_id, 
              device_map,
              rank,
              target_modules,
              output_path,
              dataset,
              seed,
              train_dataset_size,
              val_dataset_size,
              train_args)

#%%
# finetune phoenix with modified data2
train_args = {"output_dir":"./checkpoints",
            "num_train_epochs":1,
            "per_device_train_batch_size":5,
            "per_device_eval_batch_size":2,
            "gradient_accumulation_steps":2,
            "optim":'paged_adamw_32bit',
            "evaluation_strategy":'steps',
            "save_strategy":'steps',
            "load_best_model_at_end":True, # save the best model
            "metric_for_best_model":'eval_loss',
            "logging_steps":1,
            "learning_rate":2e-6,
            "weight_decay":0.001,
            "max_steps":-1,
            "warmup_ratio":0.03,
            "group_by_length":True,
            "lr_scheduler_type":"cosine",
            "gradient_checkpointing":True,
            "report_to":"none"}

model_id= "FreedomIntelligence/phoenix-inst-chat-7b"
device_map={"": 0}
rank = 8
target_modules = ["query_key_value"]
current_dir = os.getcwd()
output_path = current_dir + "/saved/phoenix/modified_data2/lora"
dataset = json.load(open('data/modified_fineval_with_common_pre.json'))
dataset = list(map(lambda sample: {"instruction": sample["instruction"], "conversations": [{"role": "human", "value": sample['input']}, {"role": "gpt", "value": sample['output']}]}, dataset))
seed = 1
train_dataset_size = 800
val_dataset_size = min(len(dataset) - train_dataset_size, train_dataset_size//4)

finetune_lora(model_id, 
              device_map,
              rank,
              target_modules,
              output_path,
              dataset,
              seed,
              train_dataset_size,
              val_dataset_size,
              train_args)

#%%
# finetune phoenix with modified data3
train_args = {"output_dir":"./checkpoints",
            "num_train_epochs":3,
            "per_device_train_batch_size":10,
            "per_device_eval_batch_size":5,
            "gradient_accumulation_steps":2,
            "optim":'paged_adamw_32bit',
            "evaluation_strategy":'steps',
            "save_strategy":'steps',
            "load_best_model_at_end":True, # save the best model
            "metric_for_best_model":'eval_loss',
            "logging_steps":1,
            "learning_rate":2e-6,
            "weight_decay":0.001,
            "max_steps":-1,
            "warmup_ratio":0.03,
            "group_by_length":True,
            "lr_scheduler_type":"cosine",
            "gradient_checkpointing":True,
            "report_to":"none"}

model_id= "FreedomIntelligence/phoenix-inst-chat-7b"
device_map={"": 0}
rank = 8
target_modules = ["query_key_value"]
current_dir = os.getcwd()
output_path = current_dir + "/saved/phoenix/modified_data3/lora"
dataset = json.load(open('data/modified_fineval_fill_blank.json'))
dataset = list(map(lambda sample: {"instruction": sample["instruction"], "conversations": [{"role": "human", "value": sample['input']}, {"role": "gpt", "value": sample['output']}]}, dataset))
seed = 1
train_dataset_size = 800
val_dataset_size = min(len(dataset) - train_dataset_size, train_dataset_size//4)

finetune_lora(model_id, 
              device_map,
              rank,
              target_modules,
              output_path,
              dataset,
              seed,
              train_dataset_size,
              val_dataset_size,
              train_args)