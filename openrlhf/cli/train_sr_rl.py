import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

# from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression, get_llm_for_rff_critic
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

# add by sR
from openrlhf.reasoning_utils.dataset import prepare_reasoning_dataset
from openrlhf.reasoning_utils.dataset import ReasoningPromptDataset
from openrlhf.datasets import SFTDataset
from openrlhf.reasoning_utils.reward_fn import setup_reward
from openrlhf.reasoning_utils.ppo_trainer import PPOTrainer
from openrlhf.models.sr_actor_critic import get_llm_for_sr_actor_critic
from openrlhf.reasoning_utils.sr_rl_trainer import SRRLTrainer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    actor, critic = get_llm_for_sr_actor_critic(
        args.pretrain, 
        "sr_actor_critic", 
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16, 
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank, 
        lora_alpha=args.lora_alpha, 
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout, 
        ds_config=strategy.get_ds_train_config(is_actor=True), 
        init_value_head=True, 
        freeze_pretrain=args.freeze_pretrain, 
        critic_hidden_size=args.critic_hidden_size, 
    )
    
    reward_model = None
    setup_reward(args.reasoning_dataset, args.cot_mode)

    # configure tokenzier
    tokenizer = get_tokenizer(args.pretrain, actor, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(actor)
    strategy.print(critic)
    
    # load weights for reference actor # CHECK: do we need to initial actor?

    if args.enable_ema:
        ema_actor, _ = get_llm_for_sr_actor_critic(
        args.pretrain, 
        "sr_actor_critic", 
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16, 
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank, 
        lora_alpha=args.lora_alpha, 
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout, 
        ds_config=strategy.get_ds_train_config(is_actor=True), 
        value_head_prefix=args.value_head_prefix, 
        init_value_head=True, 
        freeze_pretrain=args.freeze_pretrain, 
        critic_hidden_size=args.critic_hidden_size, 
        )
    else:
        ema_actor = None
    
    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        if args.enable_ema:
            ema_actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )
    
    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    # prepare datasets
    prompts_data = prepare_reasoning_dataset(
        dataset=args.reasoning_dataset, 
        cot_mode=args.cot_mode, 
        strategy=strategy, 
        seed=42, 
        max_count=args.max_samples, 
        return_eval=False, 
    )
    prompts_dataset = ReasoningPromptDataset(prompts_data, tokenizer, strategy)
    
    if args.pretrain_data:
        pretrain_data = blending_datasets(
            args.pretrain_data,
            args.pretrain_data_probs,
            strategy,
            args.seed,
            return_eval=False,
            train_split=args.pretrain_split,
        )
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )

    # prepare dataloader
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
    if args.pretrain_data:
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None
        
    # TODO: fix the max steps calculation
    update_steps_per_epoch = len(prompts_dataloader) * args.micro_rollout_batch_size // args.train_batch_size
    max_steps = update_steps_per_epoch * args.max_epochs
    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )
    critic_scheduler = None

    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        # initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        # initial_model,
        is_rlhf=True,
    )
    
    if args.enable_ema:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)
        
    # CHECK: fail safe

    os.makedirs(args.save_path, exist_ok=True)

    # configure Trainer
    trainer = SRRLTrainer(
        strategy, 
        actor, 
        critic, 
        ema_actor, 
        actor_optim, 
        critic_optim, 
        actor_scheduler, 
        critic_scheduler, 
        micro_train_batch_size=args.micro_train_batch_size, 
        micro_rollout_batch_size=args.micro_rollout_batch_size, 
        gradient_checkpointing=args.gradient_checkpointing, 
        ema_beta=0.992, 
        ptx_coef=args.ptx_coef, 
        max_epochs=args.max_epochs, 
        max_norm=args.max_norm, 
        tokenizer=tokenizer, 
        prompt_max_len=args.prompt_max_len, 
        remote_rm_url=args.remote_rm_url, 
        # for GPT generation
        do_sample=True, 
        max_new_tokens=args.generate_max_len, 
        max_length=args.max_len, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.eos_token_id, 
    )
    trainer.fit(
        args, 
        prompts_dataloader, 
        pretrain_dataloader, 
        update_steps_per_epoch
    )

    # save model checkpoint after fitting on only rank0
    strategy.save_model(
        ema_actor if args.enable_ema else actor, 
        tokenizer, 
        args.save_path
    )
    strategy.save_model(
        critic, 
        tokenizer, 
        args.save_path + "_critic"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # CHECK: below are hyper-parameters added by SR
    parser.add_argument("--freeze_pretrain", action="store_true", default=False, help="whether freeze the rff critic")
    parser.add_argument("--critic_hidden_size", type=int, default=None)
    parser.add_argument("--reasoning_dataset", type=str, default="gsm8k", help="Dataset prefix")
    parser.add_argument("--cot_mode", type=str, default="nl", help="CoT mode, nl or python_sdp")
    parser.add_argument("--update_to_data_ratio", type=int, default=1, help="Update-to-data (UTD) ratio")
    args = parser.parse_args()

    if args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain
        else:
            args.critic_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None
    train(args)
