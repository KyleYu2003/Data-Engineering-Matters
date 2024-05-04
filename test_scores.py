#%%
import os
import json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from fastchat.conversation import get_conv_template
from utilities.inference_tools import *
from utilities.scorer_tools import calculate_score
from utilities.GPT import *
#%%
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dataset = load_dataset("FinGPT/fingpt-fineval")

test_result = {}

#%%
model_id = "FreedomIntelligence/phoenix-inst-chat-7b"
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/original/lora"
model_answers = []

model_answers = generate(model_id,
                         output_path,
                         dataset['test']['instruction'],
                         dataset['test']['input'], 
                         return_answer=True)

current_dir = os.getcwd()
with open(output_path + "/test_ans_paper.json", 'w', encoding='utf-8') as writer:
    json.dump(model_answers, writer, indent=4, ensure_ascii=False)
   
test_result["phoenix_original"] = calculate_score(dataset['test']['output'], model_answers)

#%%
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/modified_data1/lora"
model_answers = []

model_answers = generate(model_id,
                         output_path,
                         dataset['test']['instruction'],
                         dataset['test']['input'], 
                         return_answer=True)

current_dir = os.getcwd()
with open(output_path + "/test_ans_paper.json", 'w', encoding='utf-8') as writer:
    json.dump(model_answers, writer, indent=4, ensure_ascii=False)
  
test_result["phoenix_modified_data1"] = calculate_score(dataset['test']['output'], model_answers)

#%% 
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/modified_data2/lora"
model_answers = [] 
    
model_answers = generate(model_id,
                         output_path,
                         dataset['test']['instruction'],
                         dataset['test']['input'], 
                         return_answer=True)

current_dir = os.getcwd()
with open(output_path + "/test_ans_paper.json", 'w', encoding='utf-8') as writer:
    json.dump(model_answers, writer, indent=4, ensure_ascii=False)

test_result["phoenix_modified_data2"] = calculate_score(dataset['test']['output'], model_answers)

#%% 
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/modified_data3/lora"
model_answers = [] 
    
model_answers = generate(model_id,
                         output_path,
                         dataset['test']['instruction'],
                         dataset['test']['input'], 
                         return_answer=True)

current_dir = os.getcwd()
with open(output_path + "/test_ans_paper.json", 'w', encoding='utf-8') as writer:
    json.dump(model_answers, writer, indent=4, ensure_ascii=False)

test_result["phoenix_modified_data3"] = calculate_score(dataset['test']['output'], model_answers)


#%%
igpt = OpenAIGPT(keys_path="/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/utilities/gpt3keys.txt")
model_answers = []
for i in range(len(dataset['test']['input'])):
    answer = igpt.call(prompt = dataset['test']['instruction'][i], user = dataset['test']['input'][i])
    model_answers.append(answer)    
calculate_score(dataset['test']['output'], model_answers)

current_dir = os.getcwd()
with open(current_dir + "/saved/GPT_ans.json", 'w', encoding='utf-8') as writer:
    json.dump(model_answers, writer, indent=4, ensure_ascii=False)
    
test_result["gpt"] = calculate_score(dataset['test']['output'], model_answers)

#%%
with open(current_dir + "/saved/GPT_ans.json", 'r', encoding='utf-8') as reader:
    model_answers = json.load(reader)

test_result["gpt"] = calculate_score(dataset['test']['output'], model_answers)