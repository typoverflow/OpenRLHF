import os
import json
from collections import defaultdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from tqdm import tqdm
import torch

def prepare_reasoning_dataset(
    dataset, 
    cot_mode, 
    tokenizer, 
    max_len, 
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
    
    def tokenize_fn(batch, max_len, tokenizer):
        assert tokenizer.eos_token_id is not None
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
        tokenize_fn, fn_kwargs={"max_len": max_len, "tokenizer": tokenizer}, 
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
            tokenize_fn, fn_kwargs={"max_len": max_len, "tokenizer": tokenizer}, 
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
        strategy, 
        pretrain_mode=False, 
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrain_mode = pretrain_mode


    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        # TODO: need to consider truncation
        full_input_ids = self.input_ids[idx] + self.answer_cot[idx]
        full_input_ids += [self.tokenzier.eos_token_id]
        return self.input_ids
        if not self.pretrain_mode:
            text = self.input_ids[idx] + self.answer_cot[idx]
        else:
            pass