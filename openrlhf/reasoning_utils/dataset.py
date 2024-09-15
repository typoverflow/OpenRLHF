import os
import json
from collections import defaultdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm
import torch

from openrlhf.datasets.utils import zero_pad_sequences

def prepare_reasoning_dataset(
    dataset, 
    cot_mode, 
    strategy=None, 
    seed=42, 
    max_count=5000000, 
    return_eval=True, 
):
    base_path = "assets/reasoning_dataset"
    train_file = f"{dataset}_nl.json"
    test_file = f"{dataset}_test_set.json"
    raw_dataset = DatasetDict({
        "train": Dataset.from_list(json.load(open(os.path.join(base_path, train_file), mode="r"))), 
        "test": Dataset.from_list(json.load(open(os.path.join(base_path, test_file), mode="r")))
    })
    
    if strategy.is_rank_0():
        strategy.print(f"Load dataset: {train_file} and {test_file}")
    
    # setup CoT
    assert dataset in {"gsm8k", "mathqa", "svamp", "mathqa-numeric"}
    assert cot_mode in {"nl", "python_sdp"}
    instruction = "Question:\n"
    cot_trigger = "\nAnswer reasoning:\n"
    answer_trigger = "\nTherefore, the answer is: "
    
    def tokenize_fn(batch):
        new_batch = defaultdict(list)
        all_keys = batch.keys()
        for item_values in zip(*(batch[k] for k in all_keys)):
            item = {k: item_values[i] for i, k in enumerate(all_keys)}
            item_id, question, answer_value, answer_cot = \
                item["item_id"], \
                item["question"], \
                item["answer_value"], \
                item.get("answer_cot", None)
            question = question.strip()
            if answer_value is not None:
                answer_value = answer_value.strip()
            if answer_cot is not None:
                answer_cot = answer_cot.strip()
                if cot_mode == "nl":
                    answer_cot += f"{answer_trigger}{answer_value}"
            
            input = f"{instruction}{question}{cot_trigger}"
            output = f"{answer_cot}"
            new_batch["prompt"].append(input)
            new_batch["answer_value"].append(answer_value)
            new_batch["answer_cot"].append(answer_cot)
            # prefix_text = f"{instruction}{question}{cot_trigger}"

            # input_encode = tokenizer(input, add_special_tokens=False)
            # output_encode = tokenizer(output, add_special_tokens=False)
            # prefix_encode = tokenizer(prefix_text, add_special_tokens=False)

            # prompt_ids = input_encode["input_ids"]
            # answer_cot_ids = output_encode["input_ids"]

            # input_ids = input_encode["input_ids"] + output_encode["input_ids"] + [tokenizer.eos_token_id]
            # labels = [-100] * len(input_ids)
            # attention_mask = [1] * len(input_ids)
            # prefix = prefix_encode["input_ids"]
            # prefix_attention_mask = prefix_encode["attention_mask"]

            # Truncation
            # prompt_ids = prompt_ids[:max_len]
            # answer_cot_ids = answer_cot_ids[:max_len]
            # input_ids = input_ids[:prompt_max_len]
            # labels = labels[:prompt_max_len]
            # attention_mask = attention_mask[:prompt_max_len]
            # prefix = prefix[:prompt_max_len]
            # prefix_attention_mask = prefix_attention_mask[:prompt_max_len]
            
            ## set new batch
            # new_batch["prompt_ids"].append(prompt_ids)
            # new_batch["labels"].append(labels)
            # new_batch["attention_mask"].append(attention_mask)
            # new_batch["prefix"].append(prefix)
            # new_batch["prefix_attention_mask"].append(prefix_attention_mask)
            ##
            # new_batch["item_id"].append(item_id)
            # new_batch["question"].append(question)
            # new_batch["prefix_text"].append(prefix_text)
            # new_batch["answer_cot_ids"].append(answer_cot_ids)
            # new_batch["answer_value"].append(answer_value)
        return new_batch
    
    train_dataset = raw_dataset["train"]
    train_dataset = train_dataset\
    .select(range(min(max_count, len(raw_dataset["train"]))))\
    .map(
        tokenize_fn, 
        batched=True,
        remove_columns=train_dataset.column_names, 
        num_proc=None, 
        load_from_cache_file=True, 
        keep_in_memory=False, 
    )
    if return_eval:
        eval_dataset = raw_dataset["test"]
        eval_dataset = eval_dataset\
        .select(range(min(max_count, len(raw_dataset["test"]))))\
        .map(
            tokenize_fn,
            batched=True,
            remove_columns=eval_dataset.column_names, 
            num_proc=None, 
            load_from_cache_file=True, 
            keep_in_memory=False,   
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


class ReasoningPromptDataset(torch.utils.data.Dataset):
    """
    Reasoning dataset for PPO model. 

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max_length of input
    """

    def __init__(
        self, 
        dataset, 
        tokenizer, 
        strategy, 
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenzier = tokenizer
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)

        self.prompt = []
        self.answer_value = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            self.prompt.append(data["prompt"])  # CHECK: add eos token?
            self.answer_value.append(data["answer_value"])

    def __len__(self):
        length = len(self.prompt)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return {
            "prompt": self.prompt[idx // self.n_samples_per_prompt],
            # "attention_mask": self.attention_masks[idx // self.n_samples_per_prompt],
            "answer_value": self.answer_value[idx // self.n_samples_per_prompt],
        }


class ReasoningSFTDataset(torch.utils.data.Dataset):
    """
    Dataset for SFT model

    """
    def __init__(
        self, 
        dataset, 
        tokenizer, 
        max_length: int, 
        strategy, 
        pretrain_mode=False, 
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        
        self.processed_dataset = dataset.map(
            self.process_data, 
            remove_columns=dataset.column_names, 
        )
        # processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

    def process_data(self, data):
        prompt, answer_value, response = data["prompt"], data["answer_value"], data["answer_cot"]
        prompt_token = self.tokenizer(prompt, add_special_tokens=False)
        prompt_ids_len = len(prompt_token["input_ids"])
        if response is None:
            input_ids = prompt_token["input_ids"] + [self.tokenizer.eos_token_id]
        else:
            response_token = self.tokenizer(response, add_special_tokens=False)
            input_ids = prompt_token["input_ids"] + response_token["input_ids"] + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        prompt_ids = prompt_token["input_ids"]
        prompt_attention_mask = prompt_token["attention_mask"]

        # truncation
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        prompt_ids = prompt_ids[:self.max_length]
        prompt_attention_mask = prompt_attention_mask[:self.max_length]

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "prompt_ids": prompt_ids, 
            "prompt_attention_mask": prompt_attention_mask, 
            "prompt_ids_len": prompt_ids_len, 
            "answer_value": answer_value, 
        }

    def __len__(self):
        length = len(self.processed_dataset)
        return length
    
    def __getitem__(self, idx):
        item = self.processed_dataset[idx]
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

