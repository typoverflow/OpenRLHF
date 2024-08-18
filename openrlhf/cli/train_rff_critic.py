import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression, get_llm_for_rff_critic
from openrlhf.trainer import RewardModelTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer
from openrlhf.trainer.rff_critic_trainer import RFFCriticTrainer

def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model/config
    rff_model = get_llm_for_rff_critic(
        args.pretrain,
        "rff_critic",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        freeze_pretrain=args.freeze_pretrain, 
        ds_config=strategy.get_ds_train_config(is_actor=True),
        init_value_head=True,
        value_head_prefix=args.value_head_prefix,
        critic_hidden_size=args.critic_hidden_size, 
    )
    actor = Actor(
        # args.pretrain, 
        rff_model, 
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    reward_model = get_llm_for_sequence_regression(
        args.reward_pretrain, 
        "reward", 
        normalize_reward=args.normalize_reward, 
        use_flash_attention_2=args.flash_attn, 
        bf16=args.bf16, 
        load_in_4bit=args.load_in_4bit, 
        ds_config=strategy.get_ds_train_config(is_actor=False), 
        value_head_prefix=args.value_head_prefix, 
    )

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, rff_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    reward_tokenzier = get_tokenizer(args.reward_pretrain, reward_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    
    
    strategy.print(actor)
    strategy.print(reward_model)
    strategy.print("reward normlization status: {}".format(args.normalize_reward))
    strategy.print("mean: {}, std: {}".format(reward_model.mean, reward_model.std))

    # configure optimizer
    optim = strategy.create_optimizer(actor, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)


    max_steps = int(len(prompts_dataloader)) \
        * (args.micro_rollout_batch_size) \
        * args.max_epochs \
        * args.num_episodes \
        // strategy.accumulated_gradient \
        // args.micro_train_batch_size 
    max_steps = math.ceil(max_steps)
    
    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )
    
    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
    
    # prepare models/optimizers...
    (
        (actor, optim, scheduler), 
        reward_model, 
    ) = strategy.prepare(
        (actor, optim, scheduler), 
        reward_model
    )

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is micro_batch_size * 2
    # we use merged chosen + rejected response forward
    trainer = RFFCriticTrainer(
        strategy,
        actor, 
        reward_model, 
        optim, 
        scheduler, 
        micro_train_batch_size=args.micro_train_batch_size, 
        micro_rollout_batch_size=args.micro_rollout_batch_size, 
        prompts_dataloader=prompts_dataloader, 
        gradient_checkpointing=args.gradient_checkpointing,
        max_epochs=args.max_epochs, 
        tokenizer=tokenizer, 
        prompt_max_len=args.prompt_max_len, 
        # for GPT generation
        do_sample=True, 
        max_new_tokens=args.generate_max_len, 
        max_length=args.max_len, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        pad_token_id=tokenizer.pad_token_id, 
        eos_token_id=tokenizer.eos_token_id
    )
    trainer.generate(args)
    trainer.fit(args)
    
    strategy.save_model(
        actor, 
        tokenizer, 
        args.save_path
    )
    
    
    # trainer = RewardModelTrainer(
    #     model=model,
    #     strategy=strategy,
    #     optim=optim,
    #     tokenizer=tokenizer,
    #     train_dataloader=train_dataloader,
    #     eval_dataloader=eval_dataloader,
    #     scheduler=scheduler,
    #     max_norm=args.max_norm,
    #     max_epochs=args.max_epochs,
    #     loss=args.loss,
    # )

    # trainer.fit(args)

    # # save model checkpoint after fitting on only rank0
    # strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 1000GB
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
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
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

    # rff critic specific
    parser.add_argument("--freeze_pretrain", action="store_true", default=True, help="whether to freeze the pretrain layers of the rff critic model.")
    parser.add_argument("--critic_hidden_size", default=512, type=int, help="hidden size of the random fourier features.")
    parser.add_argument("--learning_rate", type=float, default=9e-6)
    args = parser.parse_args()

    if args.critic_pretrain is None:
        args.critic_pretrain = args.reward_pretrain
    train(args)
