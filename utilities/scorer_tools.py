"""
This script is used to calculate the score of the model based on the multiple choice questions of fineval test.
"""
#%%
import re
import os
import json
from datasets import load_dataset

def match_choice(text: str) -> str:
    match = re.findall(r'[A-Z]', text)
    if match:
        # The answer is indicated at the beginning of response
        first_match = match[0]
        return first_match
    return ''

def calculate_score(ground_truth: str, model_answer: str) -> bool:
    count = 0
    for i in range(len(model_answer)):
        if match_choice(model_answer[i]) == match_choice(ground_truth[i]):
            count += 1
    return count/len(model_answer)

if __name__ == "__main__":
    dataset = load_dataset("FinGPT/fingpt-fineval")
    test_input = [dataset['test']['instruction'][i] + "\n\n" + dataset['test']['input'][i] for i in range(len(dataset['test']))]
    test_output = dataset['test']['output']
    current_dir = os.getcwd()
    current_dir = "/".join(current_dir.split("/")[:-1])
    with open(current_dir + '/saved/phoenix/original/lora/test_ans_paper.json', 'r') as file:
        data = json.load(file)
    print(calculate_score(test_output, data))
