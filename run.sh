#!/bin/bash
###
 # @Author: qifuxiao 867225266@qq.com
 # @Date: 2026-08-17 10:18:41
 # @FilePath: /t2v-finetune/run.sh
### 

# 指定使用 3 张 A100 (0, 1, 2)
export CUDA_VISIBLE_DEVICES=0,1,2

poetry run accelerate launch \
    --num_processes=3 \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file=ds_config.json \
    src/t2v_finetune/main.py