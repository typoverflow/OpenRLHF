deepspeed \
   --include localhost:4,5,6,7 \
   --module openrlhf.cli.train_reasoning_sft \
   --max_len 2048 \
   --train_batch_size 8 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --logging_steps 1 \
   --zero_stage 2 \
   --max_epochs 10 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --gradient_checkpointing \
   --eval_steps 256 \
   --save_steps -1 \
   --max_ckpt_num 1 \
   --reasoning_dataset gsm8k \
   --cot_mode nl \
   --max_new_tokens 1024 \
   --use_wandb 873bf35f283defae45b0d2a39312deaed163f7d6