#!/bin/bash

# 指定使用 3 张 A100 (0, 1, 2)
export CUDA_VISIBLE_DEVICES=0,1,2

poetry run accelerate launch \
    --num_processes=3 \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file=ds_config.json \
    src/t2v_finetune/train.py