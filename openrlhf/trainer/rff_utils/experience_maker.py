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
from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)

@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    # action_log_probs: torch.Tensor
    # values: torch.Tensor
    # returns: torch.Tensor
    # advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    # info: Optional[dict]
    rewards: torch.Tensor
    # info: Optional[dict]={}

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        # self.action_log_probs = self.action_log_probs.to(device)
        # self.values = self.values.to(device)
        # self.returns = self.returns.to(device)
        # self.advantages = self.advantages.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)
        self.rewards = self.rewards.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        # self.action_log_probs = self.action_log_probs.pin_memory()
        # self.values = self.values.pin_memory()
        # self.returns = self.returns.pin_memory()
        # self.advantages = self.advantages.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        self.rewards = self.rewards.pin_memory()
        return self


class RolloutExperienceMaker(ABC):
    def __init__(
        self, 
        actor: Actor, 
        reward_model: nn.Module, 
        tokenizer, 
        prompt_max_len: int, 
        strategy=None, 
        reward_fn=None
    ) -> None:
        self.actor = actor
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.strategy = strategy
        self.reward_fn = reward_fn
        
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:
        self.actor.eval()
        self.reward_model.eval()

        # generate seq
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
        num_actions = action_mask.size(1)

        # log probs
        # action_log_probs, output = self.actor(sequences, num_actions, attention_mask, return_output=True)

        # init log probs
        # base_action_log_probs = action_log_probs.clone()

        # values
        # value = output["value"]

        # rewards
        r = self.reward_model(sequences, attention_mask).unsqueeze(-1)

        # reward, kl = compute_reward(
        #     r, 
        #     self.kl_ctl.value, 
        #     action_log_probs, 
        #     base_action_log_probs, 
        #     action_mask=action_mask
        # )
        # advantage, returns = self.get_advantages_and_returns(
        #     value, 
        #     reward, 
        #     action_mask, 
        #     generate_kwargs["gamma"], 
        #     generate_kwargs["lambd"]
        # )

        # info = {
        #     "kl": masked_mean(kl, action_mask, dim=-1),
        #     "reward": r,
        #     "return": reward.sum(dim=-1),
        #     "response_length": action_mask.float().sum(dim=-1),
        #     "total_length": attention_mask.float().sum(dim=-1),
        # }

        self.actor.train()
        
        return Experience(
            sequences,
            # action_log_probs,
            # value,
            # returns,
            # advantage,
            attention_mask,
            action_mask,
            r, 
            # info,
        )
        