import os
import json
from collections import Counter
from datasets import load_dataset
from utilities.data_tools import *

# load data from FinGPT
dataset = load_dataset("FinGPT/fingpt-fineval")
# extract a label of which the exam is
labels = bind_labels(dataset["train"])
labels_counter = Counter(labels)
# select the most common labels out
most_common_labels = [t[0] for t in labels_counter.most_common(5)]
indices = [labels.index(label) for label in most_common_labels]
pre_knowledgements = dataset["train"][indices]
print(pre_knowledgements)

current_dir = os.path.dirname(__file__)

# modify the dataset
modified_dataset = dataset.map(lambda batch: concatenate_pre_knowledgement(batch, pre_knowledgements), batched=True)
modified_train_data = modified_dataset['train']

with open(current_dir + '/data/' + 'modified_fineval_with_common_pre.json', 'w') as file:
    all_data = []
    for line in modified_train_data:
        all_data.append(line)
    json.dump(all_data, file, indent=2, ensure_ascii=False)