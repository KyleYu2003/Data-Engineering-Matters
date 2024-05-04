import openai
import random
from datasets import load_dataset
from retrying import retry
from utilities.scorer_tools import calculate_score

openai.api_base = "https://api.ai-gaochao.cn/v1"


class OpenAIGPT:
    def __init__(self, model_name="gpt-3.5-turbo", keys_path=None):
        self.model_name = model_name
        with open(keys_path, encoding="utf-8", mode="r") as fr:
            self.keys = [line.strip() for line in fr if len(line.strip()) >= 4]

    def __post_process(self, response):
        return response["choices"][0]["message"]["content"]

    @retry(wait_fixed=300, stop_max_attempt_number=50)
    def call(self, prompt, user):
        current_key = self.keys[0] if len(self.keys) == 1 else random.choice(self.keys)
        openai.api_key = current_key
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            top_p=0.8,
            frequency_penalty=0.6,
            presence_penalty=0.8,
            n=1,
        )
        return self.__post_process(response)


if __name__ == "__main__":
    igpt = OpenAIGPT(keys_path="/home/zhangmin/toby/CSC6052-NLP/hw/Ass3_Kangqi/gpt3keys.txt")
    dataset = load_dataset("FinGPT/fingpt-fineval")
    model_answers = []
    for i in range(dataset['test']['input']):
        answer = igpt.call(prompt = dataset['test']['instruction'][i], user = dataset['test']['input'][i])
        model_answers.append(answer)    
    calculate_score(dataset['test']['output'], model_answers)