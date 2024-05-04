import os
import json
from datasets import load_dataset
from utilities.data_tools import *

# load data from FinGPT
dataset = load_dataset("FinGPT/fingpt-fineval")
# store the pre knowledgement before answer the question
pre_knowledgements = dataset["train"][:5]
print(pre_knowledgements)

# get current dir
current_dir = os.path.dirname(__file__)

#modify the dataset
modified_dataset = dataset.map(lambda batch: concatenate_pre_knowledgement(batch, pre_knowledgements), batched=True)
modified_train_data = modified_dataset['train']
 
with open(current_dir + '/data/' + 'modified_fineval_with_first_5_items.json', 'w') as file:
    all_data = []
    for line in modified_train_data:
        all_data.append(line)
    json.dump(all_data, file, indent=2, ensure_ascii=False)