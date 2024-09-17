import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_reward, masked_mean
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

from openrlhf.reasoning_utils.reward_fn import calculate_reward
from openrlhf.models.sr_actor_critic import CausalLMWithSROutput

logger = init_logger(__name__)


@dataclass
class Experience:
    sequences: torch.Tensor
    hidden_states: torch.Tensor
    reward: torch.FloatTensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    
    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.reward = self.reward.to(device)
        self.hidden_states = self.hidden_states.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        
    def pin_memory(self) -> None:
        self.sequences = self.sequences.pin_memory()
        self.reward = self.reward.pin_memory()
        self.hidden_states = self.hidden_states.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
            

class SRRLExperienceMaker(ABC):
    def __init__(
        self, 
        model, 
        ema_model, 
        reward_fn, 
        tokenizer, 
        prompt_max_len: int, 
        strategy=None, 
        use_ema_model: bool = False,
    ):
        super().__init__()
        self.model = model
        self.ema_model = ema_model
        self.use_ema_model = use_ema_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.strategy = strategy
        
    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}
    
    @torch.no_grad()
    def make_experience(self, prompt_ids, prompt_attention_mask, answer_value, **generate_kwargs) -> Experience:
        model_to_go = self.ema_model if self.use_ema_model else self.model
        model_to_go.eval()

        # return dicts to capture the hidden states
        # inputs = self.tokenize_fn(prompts, self.prompt_max_len, device=torch.cuda.current_device())
        sequences, attention_mask, action_mask, info = model_to_go.generate(prompt_ids, attention_mask=prompt_attention_mask, return_info=True, **generate_kwargs)
        hidden_states = info["hidden_states"]
        num_actions = action_mask.size(1)

        generated_texts = self.tokenizer.batch_decode(sequences.cpu().numpy().tolist(), skip_special_tokens=True)
        r = self.reward_fn(generated_texts, answer_value)
        # r = calculate_reward(generated_texts, answer_value)
        r = torch.tensor(r).unsqueeze(-1).to(torch.cuda.current_device()).to(hidden_states.dtype)

        info = { 
            "response_length": action_mask.float().sum(dim=-1), 
            "total_length": attention_mask.float().sum(dim=-1), 
        }
        
        model_to_go.train()
        
        return Experience(
            sequences, 
            hidden_states, 
            r, 
            attention_mask, 
            action_mask, 
            info, 
        )

    