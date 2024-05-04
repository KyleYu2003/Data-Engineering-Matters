#%%
import os
import json
from datasets import load_dataset
from utilities.data_tools import *
from utilities.scorer_tools import match_choice 

# load data from FinGPT
dataset = load_dataset("FinGPT/fingpt-fineval")
#%%
# get the correct choice
ans_list = []
for i in range(len(dataset['train']['output'])):
    ans_list.append(match_choice(dataset['train']['output'][i]))
    
#%%
input_data = []
output_data = []
for i in range(len(dataset['train']['input'])):
    input_data.append(dataset['train']['input'][i].split('\n')[0])
    output_data.append(dataset['train']['input'][i].split('\n')[1 + ord(ans_list[i]) - ord("A")])
#%%
instruction = []
for i in range(len(dataset['train']['instruction'])):
    instruction.append(dataset['train']['instruction'][i].split('单项')[0] + "填空题，请在_____上填空")
#%%
# get current dir
current_dir = os.path.dirname(__file__)

with open(current_dir + '/data/' + 'modified_fineval_fill_blank.json', 'w') as file:
    all_data = {
        "input": input_data,
        "instruction": instruction,
        "output": output_data
    }
    json.dump(all_data, file, indent=2, ensure_ascii=False)