import re
import random
import torch
from torch.utils.data import random_split

def select_random_indices(N: int, K: int) -> list:
    """Select K indices from N indices

    Args:
        N (int): Total number of indices
        K (int): Selected number of indices

    Raises:
        ValueError: K should not be greater than N

    Returns:
        list: Selected indices
    """
    if K > N:
        raise ValueError("K should not be greater than N")

    indices = list(range(N))
    selected_indices = random.sample(indices, K)

    return selected_indices

def concatenate_pre_knowledgement(data_item: dict, pre_knowledgement: dict) -> dict:
    """Concatenate pre knowledgement to the instruction

    Args:
        data_item (dict): a item in the dataset
        pre_knowledgement (dict): a item in the dataset to refer

    Returns:
        dict: a new itmem in the dataset with pre knowledgement added
    """
    instruction_head = "现在，请您扮演一位有着丰富金融市场和财务会计知识的教授，依据下面给出的练习问题并理解，完成一道考题。下面是练习问题：\n\n"
    
    count = 0
    # add pre knowledgement one by one
    while count < len(pre_knowledgement):
        instruction_head += pre_knowledgement["input"][count] + "\n" + "这道题正确答案是" + pre_knowledgement["output"][count] + "\n" + "\n"
        count += 1
    
    data_item["instruction"] = [instruction_head + "好啦，相信您已经准备充分，让我们来解决问题，请选择你认为正确的答案。\n\n" + i for i in data_item["instruction"]]
    
    return data_item

def search_pattern(text: str) -> str:
    """Using re to find the content after "关于" and the front of "的单项

    Args:
        text (str): a test string

    Returns:
        str: the content after "关于" and the front of "的单项
    """
    pattern = r'(?<=关于)(.*)(?=的单项)'
    result = re.search(pattern, text)

    if result:
        extracted_text = result.group(0).strip()
        return extracted_text
    else:
        return "未找到匹配的内容"

def bind_labels(data_item: dict) -> dict:
    """Get labels from the instruction

    Args:
        data_item (dict): a item in the dataset

    Returns:
        dict: a new itmem in the dataset with labels
    """
    label_list = [search_pattern(i) for i in data_item["instruction"]]
    return label_list

def train_test_split(seed: int, train_dataset_size: int, val_dataset_size: int, dataset: list) -> tuple:
    """Split data and return the splitted data

    Args:
        seed (int): random seed to make the training data same to different models
        train_dataset_size (int): up to user
        val_dataset_size (int): up to user
        dataset (list): a iterable object

    Returns:
        tuple: a tuple of splitted data
    """
    g = torch.Generator()
    g.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_dataset_size, val_dataset_size, len(dataset)-train_dataset_size-val_dataset_size], g)
    return train_dataset, val_dataset, test_dataset