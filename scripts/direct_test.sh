#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

# --train_llm 
# --lora_ckpt /mnt/sh/mmvision/home/changlitang/avllm/output/sft_omni_gem2.5zhCapQa_ytb13k_128f_250624/checkpoint-8000 \
# --dataset /mnt/sh/mmvision/home/changlitang/preprocess_dataset/videomme_audioVisual_short_test.json
# /mnt/sh/mmvision/home/changlitang/preprocess_dataset/librispeech_test-clean.json
#  --pred_embeds  --use_beats

bash test.sh \
    --interval 0.5 \
    --run_name debug \
    --dataset /mnt/sh/mmvision/home/changlitang/test/tmp.json \
    --max_frames 128 \
    --lora_ckpt /mnt/sh/mmvision/home/changlitang/avllm_qwenvl/output/grpo_zhEnSpkAccWERTime91-1_ctn91-500_128f_64gpuG8_selfYtbwemm1kDrama3k_oriLoraOnly_ctnRep850_260111/checkpoint-1000 \
    --model /mnt/sh/mmvision/home/changlitang/avllm_qwenvl/output/dpo_repeat_ctnTime10k128f_251224/base \
    --model_base /mnt/sh/mmvision/home/changlitang/models/Qwen3-VL-8B-Instruct-Audio \
    --gpu_num 1 --max_new_tokens 3072
