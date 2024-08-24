### TODO list of the project
+ [x] verify the process, using rff critic
+ [x] verify the use of lora -- Note: should not pass target_modules in Lora Config, not sure why
+ [ ] verify sharing the critic and the actor
+ [x] use gsm8k and math dataset
+ [x] import the corresponding reward model
+ [ ] SFT script

### Installation
```bash
pip install packaging pebble
<install pytorch 2.1.2>
pip install -e .
pip uninstall flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn
```
