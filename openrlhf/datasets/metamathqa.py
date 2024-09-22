import os
import json
from collections import defaultdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm
import torch

from openrlhf.datasets.utils import zero_pad_sequences
from openrlhf.reasoning_utils.eval import is_equiv

# prompt template
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
TRAIN_ANSWER_PREFIX = "The answer is: "
TEST_ANSWER_PREFIX = "#### "

def prepare_metamathqa_dataset(
    dataset_path, 
    mode="train", 
    strategy=None, 
    max_count=None, 
):
    dataset_name = {
        "train": "train.jsonl", 
        "eval_gsm8k": "eval_gsm8k.jsonl", 
        "eval_math": "eval_math.jsonl"
    }.get(mode)
    dataset_path = os.path.join(dataset_path, dataset_name)
    if mode == "train":
        dataset = Dataset.from_list(json.load(open(dataset_path, mode="r")))
    else:
        dataset = Dataset.from_json(dataset_path)
    if strategy.is_rank_0():
        strategy.print(f"Load MetaMathQA dataset from {dataset_path}")
    maybe_has_input = not "instruction" in dataset.column_names
    ANSWER_PREFIX = TRAIN_ANSWER_PREFIX if mode == "train" else TEST_ANSWER_PREFIX
    
    def get_input(query):
        if query.find("\n") == -1:
            return ""
        else:
            return "\n".join(query.split("\n")[1:])
    
    def apply_template(batch):
        new_batch = defaultdict(list)
        all_keys = batch.keys()
        for item in zip(*(batch.values())):
            item = {k: item[i] for i, k in enumerate(all_keys)}

            if maybe_has_input:
                instruction = item["query"].split("\n")[0]
                input = get_input(item["query"])
                response = item["response"]
            else:
                instruction = item["instruction"]
                input = ""
                response = item["response"]
            if input:
                prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
            else:
                prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
            
            prompt = prompt.strip()
            response = response.strip()
            answer = response.split(ANSWER_PREFIX)[-1]
            
            new_batch["prompt"].append(prompt)
            new_batch["response"].append(response)
            new_batch["answer"].append(answer)
        return new_batch

    if max_count is not None:
        dataset = dataset.select(range(min(max_count, len(dataset))))
    dataset = dataset.map(
        apply_template, 
        batched=True, 
        remove_columns=dataset.column_names, 
        # num_proc=4, 
        load_from_cache_file=True, 
        keep_in_memory=False
    )
    
    
    return dataset


class MetaMathQASFTDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset, 
        tokenizer, 
        max_length: int, 
        strategy, 
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        self.dataset = dataset.map(
            self.tokenize_fn, 
            remove_columns=dataset.column_names,
            num_proc=4, 
        )

    def tokenize_fn(self, data):
        prompt, response, answer = data["prompt"], data["response"], data["answer"]
        prompt_token = self.tokenizer(prompt, add_special_tokens=False)
        prompt_ids = prompt_token["input_ids"]
        prompt_ids_len = len(prompt_ids)

        response_token = self.tokenizer(response, add_special_tokens=False)
        response_ids = response_token["input_ids"]
        
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        
        attention_mask = [1] * len(input_ids)
        prompt_attention_mask = prompt_token["attention_mask"]

        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        prompt_ids = prompt_ids[:self.max_length]
        prompt_attention_mask = prompt_attention_mask[:self.max_length]
        prompt_ids_len = min(prompt_ids_len, self.max_length)

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_ids_len": prompt_ids_len,
            "answer_value": answer,
        }
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.LongTensor(item["input_ids"]),
            "attention_mask": torch.BoolTensor(item["attention_mask"]),
            "prompt_ids": torch.LongTensor(item["prompt_ids"]),
            "prompt_attention_mask": torch.BoolTensor(item["prompt_attention_mask"]),
            "prompt_ids_len": item["prompt_ids_len"], 
            "answer_value": item["answer_value"],
        }
        
    def collate_fn(self, item_list):
        # collate and pad the tokens
        # 1. basically, input_ids are prompt + response + eos, while prompt_ids are promt without eos. 
        #   if the dataset is the eval dataset without response, then input_ids is identical to prompt_ids
        # 2. apart from this, input_ids is always right padded, while prompt_ids is always left padded. 
        input_ids = []
        attention_mask = []
        prompt_ids = []
        prompt_attention_mask = []
        prompt_ids_len = []
        answer_value = []
        
        for item in item_list:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            prompt_ids.append(item["prompt_ids"])
            prompt_attention_mask.append(item["prompt_attention_mask"])
            prompt_ids_len.append(item["prompt_ids_len"])
            answer_value.append(item["answer_value"])
        
        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(attention_mask, "right")
        prompt_ids = zero_pad_sequences(prompt_ids, "left", self.tokenizer.pad_token_id)
        prompt_attention_mask = zero_pad_sequences(prompt_attention_mask, "left")

        return prompt_ids_len, input_ids, attention_mask, {
            "prompt_ids": prompt_ids, 
            "prompt_attention_mask": prompt_attention_mask, 
            "answer_value": answer_value
        }


# reward calculation
def floatify(x):
    try:
        return float(x.replace(",", ""))
    except:
        return None
    
def compare_answer(pred, target):
    if pred is None:
        return False
    if target is None:
        assert False, f"pred: {pred}, target: {target}"
    return abs(pred - target) <= 1e-2

def gsm8k_reward_fn(generated_texts, answer_values):
    pred_values = [
        floatify(res.split(TRAIN_ANSWER_PREFIX)[-1].strip()) for res in generated_texts
    ]
    target_values = [
        floatify(res.strip()) for res in answer_values
    ]
    rewards = []
    for pred, target in zip(pred_values, target_values):
        if pred is None:
            rewards.append(0.0)
        elif compare_answer(pred, target):
            rewards.append(1.0)
        else:
            rewards.append(0.1)
    return rewards
    
def gsm8k_accuracy_fn(generated_texts, answer_values):
    pred_values = [
        floatify(res.split(TRAIN_ANSWER_PREFIX)[-1].strip()) for res in generated_texts
    ]
    target_values = [
        floatify(res.strip()) for res in answer_values
    ]
    correct = [
        compare_answer(pred, target) for pred, target in zip(pred_values, target_values)
    ]
    return correct

def math_reward_fn(generated_texts, answer_values):
    pred_strings = [
        res.split(TRAIN_ANSWER_PREFIX)[-1].strip() for res in generated_texts
    ]
    target_strings = answer_values
    correct = [
        is_equiv(pred, target) for pred, target in zip(pred_strings, target_strings)
    ]
    return correct

math_accuracy_fn = math_reward_fn
     
    