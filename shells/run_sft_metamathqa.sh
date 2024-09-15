deepspeed \
   --include localhost:0,1 \
   --module openrlhf.cli.train_sft_metamathqa \
   --max_len 2048 \
   --train_batch_size 8 \
   --micro_train_batch_size 4 \
   --max_samples 500000 \
   --pretrain google/gemma-2b-it \
   --save_path ./checkpoint/gemma-2b-it-sft-metamathqa/save \
   --ckpt_path ./checkpoint/gemma-2b-it-sft-metamathqa \
   --logging_steps 1 \
   --zero_stage 2 \
   --max_epochs 5 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-5 \
   --gradient_checkpointing \
   --eval_steps -1 \
   --save_steps -1 \
   --max_ckpt_num 5 \
   --max_new_tokens 1024 \
   --use_wandb 873bf35f283defae45b0d2a39312deaed163f7d6