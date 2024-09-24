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
        ema_critic: nn.Module, 
        actor_optim: Optimizer, 
        critic_optim: Optimizer, 
        actor_scheduler: Any, 
        critic_scheduler: Any, 
        critic_beta: float = 0.995, 
        actor_beta: float = 0.995, 
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
        accuracy_fn: Any = None, 
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
        self.actor_beta = actor_beta
        self.critic_beta = critic_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn
        self.accuracy_fn = accuracy_fn

        self.actor = actor
        self.critic = critic
        self.ema_actor = ema_actor
        self.ema_critic = ema_critic
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
        eval_dataloader, 
        pretrain_dataloader, 
        num_update_steps_per_epoch=1, 
        # consumed_samples=0, 
        # num_update_steps_per_episodes=1, 
    ) -> None:
        # micro_rollout_steps is the total number of step for micro rollout to generate rollout_batch_size samples
        self.micro_rollout_steps = args.rollout_batch_size // (self.strategy.world_size * args.micro_rollout_batch_size)
        # micro_update_steps is the total number of micro updates to perform one gradient step
        self.micro_update_steps = args.train_batch_size // (self.strategy.world_size * args.micro_train_batch_size)
        # update_steps is the total number of update steps to satisfy UTD
        self.update_to_data_ratio = args.update_to_data_ratio
        self.update_steps = args.rollout_batch_size // args.train_batch_size * args.update_to_data_ratio
        self.freeze_actor_steps = args.freeze_actor_steps

        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = num_update_steps_per_epoch
        elif args.save_steps == 0:
            args.save_steps = float("inf")
            
        # data loaders
        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        self.tot_rollout_micro_steps = 0
        self.tot_update_micro_steps = 0
        start_epoch = 0
        consumed_samples = 0
        self.evaluate(eval_dataloader, global_step=0)  # first evaluation to generate reference score
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
            
            for _, __, ___, info in self.prompts_dataloader:
                prompt_ids = info["prompt_ids"].to(torch.cuda.current_device())
                prompt_attention_mask = info["prompt_attention_mask"].to(torch.cuda.current_device())
                experience = self.experience_maker.make_experience(
                    prompt_ids=prompt_ids, 
                    prompt_attention_mask=prompt_attention_mask, 
                    answer_value=info["answer_value"], 
                    **self.generate_kwargs
                )
                self.replay_buffer.append(experience)
                self.tot_rollout_micro_steps += 1
                
                if (self.tot_rollout_micro_steps % self.micro_rollout_steps == 0):
                    # samples are ready, start update
                    output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
                    self.strategy.print(output[0])
                    
                    status_mean = {}  # stores the average statistics during one update
                    status_list = []
                    torch.cuda.empty_cache()
                    for _ in range(self.update_steps * self.micro_update_steps):
                        experience = self.replay_buffer.sample()
                        status = {}
                        status.update(self.training_step_actor(experience))
                        status.update(self.training_step_critic(experience))
                        status_list.append(status)
                        self.tot_update_micro_steps += 1

                        # evaluation and checkpoint saving
                        if self.tot_update_micro_steps % self.strategy.accumulated_gradient == 0:
                            self.save_logs_and_checkpoint(args, self.tot_update_micro_steps//self.strategy.accumulated_gradient, status, pbar)
                                
                    torch.cuda.empty_cache()
                    
                    status_mean["act_loss"] = sum([s["actor_loss"] for s in status_list if "actor_loss" in s])
                    status_mean["cri_loss"] = sum([s["critic_loss"] for s in status_list if "critic_loss" in s])
                    status_mean["act_loss"] /= self.update_steps * self.micro_update_steps
                    status_mean["cri_loss"] /= self.update_steps * self.micro_update_steps
                    pbar.set_postfix(status_mean)

                pbar.update()
                
    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        if self.tot_update_micro_steps < self.freeze_actor_steps * self.micro_update_steps:
            return {}
        self.actor.train()
        self.critic.requires_grad_(False)  # this is importance because deepspeed do not support differentiate through multiple engines that require gradients
        
        results = self.actor(experience.sequences, experience.attention_mask)
        hidden_states = results.roll_reps[:, -experience.action_mask.size(1):]  # select the current hidden states
        action_mask = experience.action_mask
        # action_mask = experience.action_mask[:, :-1]
        actor_loss = - self.critic(hidden_states)
        actor_loss = masked_mean(actor_loss, action_mask, dim=-1).mean()

        # self.strategy.zero_grad(self.actor_optim, self.actor)  # actually useless in deepspeed
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
            if self.critic_beta is not None:
                all_target_values = self.ema_critic(all_hidden_states)
            else:
                all_target_values = self.critic(all_hidden_states)
            # target_values = self.critic(next_hidden_states)

            # determine where to insert the reward
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
            all_reward = torch.zeros_like(all_target_values).scatter_(dim=1, index=eos_indices, src=experience.reward)
            
            target_values = all_reward[:, 1:] + action_mask[:, 1:] * all_target_values[:, 1:]
            # target_values[:, -1] = experience.reward  # set the target of last time step to reward
        critic_loss = 0.5 * (target_values - values).pow(2)
        critic_loss = masked_mean(critic_loss, action_mask[:, :-1], dim=-1).mean()

        # self.strategy.zero_grad(self.critic_optim, self.critic)  # this is actually useless
        self.strategy.backward(critic_loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")
        if self.critic_beta is not None:
            self.strategy.moving_average(self.critic, self.ema_critic, self.critic_beta, self.ema_critic.device)        
        self.critic.eval()
        status = {
            "critic_loss": critic_loss.item(), 
            "values": masked_mean(values, action_mask[:, :-1]).item(), 
            "critic_lr": self.critic_scheduler.get_last_lr()[0] if self.critic_scheduler is not None else 0.0
        }
        return status

    def save_logs_and_checkpoint(self, args, global_step, status, pbar):
        if global_step % args.logging_steps == 0:
            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**status, "global_step": global_step}.items()}
                self._wandb.log(logs)
                
        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_model(
                self.actor, 
                self.tokenizer, 
                os.path.join(args.save_path, tag)
            )
            # self.strategy.save_ckpt(
            #     self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, {}    
            # )
        
    def evaluate(self, eval_dataloader, global_step=0):
        times = 0
        self.actor.eval()
        self.generate_kwargs["do_sample"] = False  # use greedy decoding during evaluation
        with torch.no_grad():
            acc_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % global_step,
                disable=not self.strategy.is_rank_0(),
            )

            for prompts_id_lens, inputs, attention_masks, infos in eval_dataloader:
                # generation
                prompt_ids = infos["prompt_ids"].to(torch.cuda.current_device())
                prompt_attention_mask = infos["prompt_attention_mask"].to(torch.cuda.current_device())
                sequences, attention_mask, action_mask = self.actor.generate(
                    input_ids=prompt_ids, 
                    attention_mask=prompt_attention_mask, 
                    **self.generate_kwargs
                )
                generated_texts = self.tokenizer.batch_decode(sequences.cpu().numpy().tolist(), skip_special_tokens=True)
                acc = self.accuracy_fn(generated_texts, infos["answer_value"])

                times += 1
                acc_sum += sum(acc) / len(acc)
                bar_dict = {"eval acc": acc_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)
                
            ### print the results
            if self.strategy.is_rank_0():
                self.strategy.print(f"######### Evaluation #########")
                for _ in range(min(len(generated_texts), 4)):
                    self.strategy.print(f"Reward: {acc[_]}, Text: {generated_texts[_]}")
                    self.strategy.print(f"##############")
            ###

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
        self.generate_kwargs["do_sample"] = True  # reset to default
        self.actor.train()  # reset model state

