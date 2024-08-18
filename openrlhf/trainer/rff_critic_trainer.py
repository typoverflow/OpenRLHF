import math
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean

# from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveReplayBuffer, RolloutExperienceMaker
from .rff_utils import Experience, RolloutReplayBuffer, RolloutExperienceMaker


class RFFCriticTrainer(ABC):
    def __init__(
        self, 
        strategy, 
        actor: Actor, 
        reward_model: nn.Module, 
        optim: Optimizer, 
        scheduler, 
        micro_train_batch_size: int = 8, 
        buffer_limit: int = 0, 
        buffer_cpu_offload: bool = True, 
        micro_rollout_batch_size: int = 8, 
        prompts_dataloader=None, 
        gradient_checkpointing: bool = False, 
        max_epochs: int = 1, 
        tokenizer: Optional[Callable[[Any], dict]] = None, 
        prompt_max_len: int = 128, 
        dataloader_pin_memory: bool = True, 
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None, 
        **generate_kwargs, 
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.prompts_dataloader = prompts_dataloader
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.prompt_max_len = prompt_max_len
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.reward_model = reward_model
        self.optim = optim
        self.scheduler = scheduler

        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.experience_maker = RolloutExperienceMaker(
            actor, reward_model, tokenizer, prompt_max_len, strategy, reward_fn
        )

        self.replay_buffer = RolloutReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rand_0():
            import wandb
            
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

    def generate(self, args) -> None:
        steps = 0
        for episode in range(args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )
            for rand_prompts in self.prompts_dataloader:
                experience = self.experience_maker.make_experience(rand_prompts, **self.generate_kwargs)
                self.replay_buffer.append(experience)

                pbar.update()
                steps += 1
                self.strategy.print(steps)
                if steps >= 10:
                    return
    
    def save_generate(self):
        raise NotImplementedError
    
    def load_generate(self):
        raise NotImplementedError
    
    def fit(self, args):
        dataloader = DataLoader(
            self.replay_buffer, 
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True, 
            drop_last=True, 
            pin_memory=self.dataloader_pin_memory, 
            collate_fn=self.replay_buffer.collate_fn
        )
        device = torch.cuda.current_device()

        global_step = 1
        epoch_bar = tqdm(range(self.max_epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.max_epochs):
            step_bar = tqdm(
                range(dataloader.__len__()), 
                desc="Train step of epoch {}".format(epoch), 
                disable=not self.strategy.is_rank_0()
            )
            if isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
                
            self.actor.train()
            loss_mean = 0.
            for experience in dataloader:
                experience.to_device(device)
                output = self.actor.forward_with_value(
                    experience.sequences, 
                    attention_mask=experience.attention_mask, 
                    return_output=True
                )
                value_preds = output["values"]
                
                with torch.no_grad():
                    value_targets = torch.concat([
                        value_preds[:, 1:], 
                        experience.rewards.unsqueeze(1)
                    ], dim=1)
                value_loss = 0.5 * (value_preds - value_targets).pow(2)
                value_loss = (value_loss * experience.action_masks).sum(1) / experience.action_masks.sum(1)

                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                
                loss = value_loss + aux_loss * self.args.aux_loss_coef            
                self.strategy.backward(loss, self.actor, self.optim)
                self.strategy.optimizer_step(self.optim, self.actor, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()

                logs_dict = {
                    "value_loss": loss.item(), 
                    "value_pred": value_preds.mean().item(), 
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                self.save_logs_and_checkpoints(self.args, global_step, step_bar, logs_dict)
                
                step_bar.update()
                global_step += 1
            epoch_bar.update()
        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
            
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
        
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.actor, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)
