#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

export CUDA_VISIBLE_DEVICES=0

bash demo.sh \
    --interval 0.5 \
    --max_frames 128 \
    --lora_ckpt /mnt/sh/mmvision/home/changlitang/avllm_qwenvl/output/grpo_zhEnSpkAccWERTime91-1_ctn91-500_128f_64gpuG8_selfYtbwemm1kDrama3k_oriLoraOnly_ctnRep850_260111/checkpoint-1000 \
    --model /mnt/sh/mmvision/home/changlitang/avllm_qwenvl/output/dpo_repeat_ctnTime10k128f_251224/base \
    --model_base /mnt/sh/mmvision/home/changlitang/models/Qwen3-VL-8B-Instruct-Audio \
