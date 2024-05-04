#%%
import json
import re
from utilities.GPT import *

igpt = OpenAIGPT(keys_path="/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/utilities/gpt3keys.txt")

prompt = """
We would like to request your feedback on the two AI assistants in response to the user question displayed above. \n
Please evaluate the helpfulness, relevance, accuracy, level of details of their responses. You should tell me whether Assistant 1 is `better than`, `worse than`, or `equal to` Assistant 2. \n
Please first compare their responses and analyze which one is more in line with the given requirements. \n
In the last line, please output a single line containing only a single label selecting from `Assistant 1 is better than Assistant 2`, `Assistant 1 is worse than Assistant 2`, and `Assistant 1 is equal to Assistant 2`, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
"""

test_file = "zh_fin.json"

with open("./test_dataset/zh_fin.json", 'r', encoding='utf-8') as reader:
    test_data = json.load(reader)
print(test_data[0])

#%%
# the model fine-tuned by the orignal dataset
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/original/lora"
with open(output_path + "/saved_data.json", 'r', encoding='utf-8') as reader:
    model_ans = json.load(reader)

model_response = []
for i in range(len(test_data)):
    response = igpt.call(prompt = prompt, user = "Question: " + test_data[i][0] + '\n' + "Assistant1: " + test_data[i][1] + "Assistant2: " + model_ans[i][2])
    model_response.append(response) 

gpt_win_count = 0
for i in range(len(model_response)):
    match = re.search(r"(?<=Assistant\s)(.*?)(?=\sis better than)", model_response[i])
    if match:
        print(match)
        if match[0][-1].strip() == '1':
            gpt_win_count += 1
print(gpt_win_count)

#%%
# the model fine-tuned by the modified dataset1
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/modified_data1/lora"
with open(output_path + "/saved_data.json", 'r', encoding='utf-8') as reader:
    model_ans = json.load(reader)

model_response = []
for i in range(len(test_data)):
    response = igpt.call(prompt = prompt, user = "Question: " + test_data[i][0] + '\n' + "Assistant1: " + test_data[i][1] + "Assistant2: " + model_ans[i][2])
    model_response.append(response) 

gpt_win_count = 0
for i in range(len(model_response)):
    match = re.search(r"(?<=Assistant\s)(.*?)(?=\sis better than)", model_response[i])
    if match:
        print(match)
        if match[0][-1].strip() == '1':
            gpt_win_count += 1
print(gpt_win_count)

#%%
# the model fine-tuned by the modified dataset2
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/modified_data2/lora"
with open(output_path + "/saved_data.json", 'r', encoding='utf-8') as reader:
    model_ans = json.load(reader)

model_response = []
for i in range(len(test_data)):
    response = igpt.call(prompt = prompt, user = "Question: " + test_data[i][0] + '\n' + "Assistant1: " + test_data[i][1] + "Assistant2: " + model_ans[i][2])
    model_response.append(response) 

gpt_win_count = 0
for i in range(len(model_response)):
    match = re.search(r"(?<=Assistant\s)(.*?)(?=\sis better than)", model_response[i])
    if match:
        print(match)
        if match[0][-1].strip() == '1':
            gpt_win_count += 1
print(gpt_win_count)

#%%
# the model fine-tuned by the modified dataset3
output_path = "/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/saved/phoenix/modified_data3/lora"
with open(output_path + "/saved_data.json", 'r', encoding='utf-8') as reader:
    model_ans = json.load(reader)

model_response = []
for i in range(len(test_data)):
    response = igpt.call(prompt = prompt, user = "Question: " + test_data[i][0] + '\n' + "Assistant1: " + test_data[i][1] + "Assistant2: " + model_ans[i][2])
    model_response.append(response) 

gpt_win_count = 0
for i in range(len(model_response)):
    match = re.search(r"(?<=Assistant\s)(.*?)(?=\sis better than)", model_response[i])
    if match:
        print(match)
        if match[0][-1].strip() == '1':
            gpt_win_count += 1
print(gpt_win_count)