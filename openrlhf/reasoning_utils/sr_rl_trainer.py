import math
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController, NaiveReplayBuffer
# from .ppo_utils.experience_maker import Experience, NaiveExperienceMaker
from openrlhf.models.actor_critic import CausalLMWithRFFCriticOutputWithPast

from .rl_utils.experience_maker import Experience, SRRLExperienceMaker
from .rl_utils.replay_buffer import SRRLReplayBuffer

class SRRLTrainer(ABC):
    def __init__(
        self, 
        strategy, 
        actor: nn.Module, 
        critic: nn.Module, 
        ema_actor: nn.Module, 
        actor_optim: Optimizer, 
        critic_optim: Optimizer, 
        actor_scheduler: Any, 
        critic_scheduler: Any, 
        ema_beta: float = 0.992, 
        ptx_coef: float = 0.0, 
        micro_train_batch_size: int = 8, 
        buffer_limit: int = 0, 
        buffer_cpu_offload: bool = True, 
        micro_rollout_batch_size: int = 2, 
        gradient_checkpointing: bool = False, 
        max_epochs: int = 1, 
        max_norm: float = 1.0, 
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None, 
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args

        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.ema_actor = ema_actor
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        
        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8
        
        self.experience_maker = SRRLExperienceMaker(
            model=actor, 
            ema_model=ema_actor, 
            reward_fn=reward_fn, 
            tokenizer=tokenizer, 
            prompt_max_len=prompt_max_len, 
            strategy=strategy, 
            use_ema_model=False, # TODO: determine whether to use ema model
        )
        self.replay_buffer = SRRLReplayBuffer(
            sample_batch_size=self.micro_rollout_batch_size, 
            limit=buffer_limit, 
            cpu_offload=buffer_cpu_offload, 
        )

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
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
            
    def fit(
        self, 
        args, 
        prompts_dataloader, 
        pretrain_dataloader, 
        # consumed_samples=0, 
        # num_update_steps_per_episodes=1, 
    ) -> None:
        update_interval = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)
        update_steps = args.rollout_batch_size // (self.strategy.world_size * self.micro_train_batch_size)

        
        
        # num_rollouts_per_episodes = (
        #     num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size
        # )
        # # make sure UTD = 1
        # update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_train_batch_size)
        
        # # get eval and save steps
        # if args.eval_steps == -1:
        #     args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        # if args.save_steps == -1:
        #     args.save_steps = float("inf")  # do not save ckpt
            
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        # steps = consumed_samples // args.rollout_batch_size * update_timesteps + 1
        # start_epoch = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        # consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)
        steps = 0
        start_epoch = 0
        consumed_samples = 0
        for epoch in range(start_epoch, args.max_epochs):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Epoch [{epoch + 1}/{args.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            
            for rand_prompts in self.prompts_dataloader:
                experience = self.experience_maker.make_experience(
                    prompts=rand_prompts["prompt"], 
                    answer_values=rand_prompts["answer_value"], 
                    **self.generate_kwargs
                )
                self.replay_buffer.append(experience)
                if (steps+1) % update_interval == 0:
                    # print the output
                    output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
                    self.strategy.print(output[0])

                    # train for several steps
                    global_steps = steps // update_interval
                    torch.cuda.empty_cache()
                    status = self.rl_train(global_steps, update_steps)
                    torch.cuda.empty_cache()

                    if "kl" in status:
                        self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    pbar.set_postfix(status)
                    
                    # logs/checkpoints
                    # client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                    # self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1
                
    def rl_train(self, global_steps=0, update_steps=1):
        # dataloader = DataLoader(
        #     self.replay_buffer, 
        #     batch_size=self.replay_buffer.sample_batch_size, 
        #     shuffle=True, 
        #     drop_last=True, 
        #     pin_memory=self.dataloader_pin_memory, 
        #     collate_fn=self.replay_buffer.collate_fn, 
        # )
        # dataloader = iter(dataloader)

        status_list = []
        status_mean = {}
        for step in tqdm(range(update_steps), desc="Train step"):
            # experience = next(dataloader)
            experience = self.replay_buffer.sample()
            
            this_status = {}
            if global_steps > self.freezing_actor_steps:
                this_status = self.training_step_actor(experience)
            this_status.update(self.training_step_critic(experience))

            # TODO save the statistics

        return status_mean

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        self.critic.requires_grad_(False)  # this is importance because deepspeed do not support differentiate through multiple engines that require gradients
        
        results = self.actor(experience.sequences, experience.attention_mask)
        hidden_states = results.roll_reps[:, -experience.action_mask.size(1):]  # select the current hidden states
        action_mask = experience.action_mask
        # action_mask = experience.action_mask[:, :-1]
        actor_loss = - self.critic(hidden_states)
        actor_loss = masked_mean(actor_loss, action_mask, dim=-1).mean()

        self.strategy.backward(actor_loss, self.actor, self.actor_optim)
        
        # TODO add mask
        # TODO add ptx loss
        # TODO ptx loss

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        # TODO ema update
        
        self.critic.requires_grad_(True)
        self.actor.eval()
        
        status = {
            "policy_loss": actor_loss.item(), 
            "actor_lr": self.actor_scheduler.get_last_lr()[0]
        }
        return status


    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()
        
        values = self.critic(experience.hidden_states[:, :-1]) # select the current hidden states
        action_mask = experience.action_mask
        with torch.no_grad():
            results = self.actor(experience.sequences, experience.attention_mask)
            all_hidden_states = results.roll_reps[:, -experience.action_mask.size(1):]
            # next_hidden_states = results.roll_reps[:, -experience.action_mask.size(1)+1:] # select the next hidden states
            all_target_values = self.critic(all_hidden_states)
            # target_values = self.critic(next_hidden_states)

            # determine where to insert the reward
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
            all_reward = torch.zeros_like(all_target_values).scatter_(dim=1, index=eos_indices, src=experience.reward)
            
            target_values = all_reward[:, 1:] + action_mask[:, 1:] * all_target_values[:, 1:]
            # target_values[:, -1] = experience.reward  # set the target of last time step to reward
        critic_loss = 0.5 * (target_values - values).pow(2)
        critic_loss = masked_mean(critic_loss, action_mask[:, :-1], dim=-1).mean()

        self.strategy.zero_grad(self.critic_optim, self.critic)
        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        
        self.critic.eval()
        status = {
            "critic_loss": critic_loss.item(), 
            "values": masked_mean(values, action_mask[:, :-1]).item(), 
            "critic_lr": self.critic_scheduler.get_last_lr()[0] if self.critic_scheduler is not None else 0.0
        }
        return status

