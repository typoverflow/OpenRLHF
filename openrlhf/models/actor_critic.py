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
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

logger = init_logger(__name__)

@dataclass
class CausalLMWithRFFCriticOutputWithPast(CausalLMOutputWithPast):
    values: Optional[torch.FloatTensor] = None
    

def get_llm_for_rff_actor_critic(
    model_name_or_path: str, 
    model_type: str,   # must be rff_critic
    *, 
    bf16=True, 
    load_in_4bit=False, 
    lora_rank=0, 
    lora_alpha=16, 
    target_modules=None, 
    lora_dropout=0, 
    normalize_reward=False,
    use_flash_attention_2=False, 
    ds_config: dict = None, 
    init_value_head: bool = False, 
    value_head_prefix="value_head", 
    device_map=None, 
    packing_samples=False, 
    freeze_pretrain=True, 
    critic_hidden_size=512, 
    **kwargs, 
) -> nn.Module:
    """Get transformer with a rff value head on top
    """
    assert model_type == "rff_actor_critic", f"model type must be rff_actor_critic"
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    config.critic_hidden_size = critic_hidden_size

    base_class = AutoModelForCausalLM._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    
    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    class CausalLMWithRFFCritic(base_class):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, 
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.critic_hidden_size), 
                    Sin(), 
                    nn.Linear(config.critic_hidden_size, config.critic_hidden_size), 
                    nn.ReLU(), 
                    nn.Linear(config.critic_hidden_size, 1)
                )        
            )
            
            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std
                
            # getattr(self, self.base_model_prefix).__original_forward = getattr(self, self.base_model_prefix).forward
            # getattr(self, self.base_model_prefix).__original_prepare_inputs_for_generation = getattr(self, self.base_model_prefix).prepare_inputs_for_generation
            # getattr(self, self.base_model_prefix).forward = self.__modified_base_model_forward
            # getattr(self, self.base_model_prefix).prepare_inputs_for_generation = self.__modified_base_model_prepare

        # def __modified_base_model_prepare(
        #     self, 
        #     base_model, 
        #     input_ids, 
        #     past_key_values=None, 
        #     attention_mask=None, 
        #     inputs_embeds=None, 
        #     **kwargs
        # ):
        #     past_key_values_0, past_key_values_1 = past_key_values if past_key_values is not None else (None, None)
        #     inputs_embeds_0, inputs_embeds_1 = inputs_embeds if inputs_embeds is not None else (None, None)
        #     inputs = base_model.__original_prepare_inputs_for_generation(input_ids, past_key_values=past_key_values_0, attention_mask=attention_mask, inputs_embeds=inputs_embeds_0, **kwargs)
        #     if "past_key_values" in inputs:
        #         inputs["past_key_values"] = (past_key_values_0, past_key_values_1)
        #     if "inputs_embeds" in inputs:
        #         inputs["inputs_embeds"] = (inputs_embeds_0, inputs_embeds_1)
        #     return inputs
        
        def forward(
            self, 
            input_ids: torch.LongTensor = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            modify_representation: bool=True, 
            action_mask: Optional[torch.Tensor] = None, 
            **kwargs
        ):
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            kwargs["output_hidden_states"] = True  # we need hidden states to modify the logits
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            last_hidden_states = outputs["hidden_states"][-1]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)[:, :-1]
            if modify_representation:
                # modify both logits and log_probs
                pass
            else:
                pass
            
            if action_mask is not None:
                num_actions = action_mask.size(1)
                values = values[:, -num_actions:]

            return CausalLMWithRFFCriticOutputWithPast(
                loss=outputs.loss, 
                logits=outputs.logits, 
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                values=values
            )
    
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
        
    model = CausalLMWithRFFCritic.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )
    
    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # freeze the base model
    # should be used when LoRA is not enabled
    if freeze_pretrain:
        for pname, p in model.named_parameters():
            if "value_head" not in pname:
                p.requires_grad = False
                
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
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head[0].weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            model.value_head[2].weight.data.normal_(mean=0.0, std=1 / (config.critic_hidden_size + 1))
            model.value_head[4].weight.data.normal_(mean=0.0, std=1 / (config.critic_hidden_size + 1))

    return model
            
        # def __modified_base_model_forward(
        #     self, 
        #     base_model, 
        #     input_ids=None, 
        #     attention_mask=None, 
        #     inputs_embeds=None, 
        #     labels=None, 
        #     output_attentions=None, 
        #     output_hidden_states=None, 
        #     return_dict=None, 
        #     modify_reprensetation=True, 
        #     **kwargs
        # ):
        #     kwargs.update(
        #         {
        #             "attention_mask": attention_mask, 
        #             "labels": labels, 
        #             "output_attentions": output_attentions, 
        #             "output_hidden_states": output_hidden_states, 
        #             "return_dict": return_dict
        #         }
        #     )
        #     base_model_output = base_model.__original_forward(
        #         input_ids=input_ids, 
        #         inputs_embeds=inputs_embeds, 
        #         **kwargs
        #     )
        #     # TODO
        #     return CausalLMWithRFFCriticOutputWithPast(
        #         logits=modified_logits, 
        #         past_key_values=base_model_output.past_key_values, 
        #         hidden_states=base_model_output.hidden_states, 
        #         attentions=base_model_output.attentions, 
        #     )