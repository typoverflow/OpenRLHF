from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .packing_utils import patch_for_block_diag_attn
from .utils import reset_position_ids
from openrlhf.utils.logging_utils import init_logger

from openrlhf.models.actor import Actor
from trl import PreTrainedModelWrapper
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

logger = init_logger(__name__)


@dataclass
class CausalLMWithSROutput(CausalLMOutputWithPast):
    base_reps: Optional[torch.FloatTensor] = None
    roll_reps: Optional[torch.FloatTensor] = None
    
def get_llm_for_sr_actor_critic(
    model_name_or_path: str, 
    model_type: str,   # must be rff_critic
    *, 
    bf16=True, 
    load_in_4bit=False, 
    lora_rank=0, 
    lora_alpha=16, 
    target_modules=None, 
    lora_dropout=0, 
    use_flash_attention_2=False, 
    ds_config: dict = None, 
    init_value_head: bool = False, 
    device_map=None, 
    packing_samples=False, 
    freeze_pretrain=True, 
    critic_hidden_size=None, 
    **kwargs, 
) -> nn.Module:
    """
    Get a transformer model with a critic head
    """
    assert model_type == "sr_actor_critic"
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    if critic_hidden_size is None:
        critic_hidden_size = config.hidden_size
    config.critic_hidden_size = critic_hidden_size

    base_class = AutoModelForCausalLM._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__

    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    class CausalLMWithSR(base_class):
        supports_gradient_checkpointing = True
        
        def __init__(self, config: AutoConfig):
            super().__init__(config)
            self.packing_samples = packing_samples
                
        def forward(
            self, 
            input_ids: torch.LongTensor = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            **kwargs
        ):
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            kwargs["output_hidden_states"] = True  # need hidden states for SR
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            last_hidden_states = outputs["hidden_states"][-1]
            return CausalLMWithSROutput(
                loss=outputs.loss, 
                logits=outputs.logits, 
                past_key_values=outputs.past_key_values, 
                hidden_states=outputs.hidden_states, 
                attentions=outputs.attentions, 
                base_reps=None, 
                roll_reps=last_hidden_states
            )
        
        @torch.no_grad()
        def generate(self, input_ids: torch.Tensor, return_info: bool=False, **kwargs):
            generate_args = {
                "input_ids": input_ids,
                "top_k": kwargs.get("top_k", None),
                "top_p": kwargs.get("top_p", None),
                "do_sample": kwargs.get("do_sample", True),
                # "early_stopping": True,  # CHECK: remember to add this when num_beams > 1
                "temperature": kwargs.get("temperature", 1),
                "use_cache": True,
                "num_beams": kwargs.get("num_beams", 1),
                "attention_mask": kwargs.get("attention_mask"),
                "eos_token_id": kwargs.get("eos_token_id"),
                "pad_token_id": kwargs.get("pad_token_id"),
                "min_new_tokens": kwargs.get("min_new_tokens", 1),
                # added for sr, return the output hidden states
                "return_dict_in_generate": True, 
                "output_hidden_states": True
            }

            if kwargs.get("max_new_tokens", None):
                generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
            if kwargs.get("max_length", None):
                generate_args["max_length"] = kwargs.get("max_length")

            # Call generate
            # sequences = self.model.generate(**generate_args)
            results = super().generate(**generate_args)

            # Prepare mask tensor
            eos_token_id = generate_args["eos_token_id"]
            pad_token_id = generate_args["pad_token_id"]

            return self.process_sequences(results, input_ids.size(1), eos_token_id, pad_token_id, return_info=return_info)

        def process_sequences(self, results, input_len, eos_token_id, pad_token_id, return_info=False):
            sequences = results.sequences
            attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
            seq_length = attention_mask.size(1)

            # The following code is equivalent to:
            #
            # for i in range(attention_mask.size(0)):
            #     for t in reversed(range(seq_length)):
            #         if attention_mask[i][t] > 0.5:
            #             attention_mask[i][min(t + 1, seq_length - 1)] = True
            #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
            #             break
            #
            eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
            sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

            # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
            first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
            mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
            attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

            # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
            state_seq = sequences[:, input_len - 1 : -1]
            action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
            action_mask[:, 0] = 1
            
            # organize the output info
            if return_info:
                info = {}
                hidden_states = results.hidden_states
                hidden_states = torch.concat([h[-1] for h in hidden_states], dim=1)
                hidden_states = hidden_states[:, input_len - 1 :]  # select the state part hidden states
                info["hidden_states"] = hidden_states
            else:
                info = None
            
            if return_info:
                return sequences, attention_mask, action_mask, info
            else:
                return sequences, attention_mask, action_mask 
    
    class CriticWithSR(nn.Module):
        def __init__(self, hidden_size, critic_hidden_size):
            super().__init__()
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, critic_hidden_size), 
                Sin(), 
                nn.Linear(critic_hidden_size, critic_hidden_size), 
                nn.ReLU(), 
                nn.Linear(critic_hidden_size, 1)
            )
            # TODO no normalize reward

        def forward(
            self, 
            input: torch.FloatTensor
        ):
            value = self.value_head(input)
            return value.squeeze(-1)
    
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
        
    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None
        
    model = CausalLMWithSR.from_pretrained(
        model_name_or_path, 
        config=config, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16 if bf16 else "auto", 
        quantization_config=nf4_config, 
        device_map=device_map, 
        **kwargs
    )
    
    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank, 
            lora_alpha=lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=lora_dropout, 
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        
        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "value_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)
    
    # should not freeze pretrain
    assert not freeze_pretrain

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True
        
    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False
    
    # packing samples using Flash Attention 2
    if packing_samples:
        assert use_flash_attention_2, "Only support `--packing_samples` with Flash Attention 2."
        model_type = getattr(model.config, "model_type", None)
        patch_for_block_diag_attn(model_type)
        
    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    critic = CriticWithSR(config.hidden_size, config.critic_hidden_size)
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    critic.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            critic.value_head[0].weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            critic.value_head[2].weight.data.normal_(mean=0.0, std=1 / (config.critic_hidden_size + 1))
            critic.value_head[4].weight.data.normal_(mean=0.0, std=1 / (config.critic_hidden_size + 1))

    return model, critic

# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# a = LlamaForCausalLM.generate