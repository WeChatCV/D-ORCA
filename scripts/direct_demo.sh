#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

export CUDA_VISIBLE_DEVICES=0

bash demo.sh \
    --interval 0.5 \
    --max_frames 128 \
    --model /mnt/sh/mmvision/home/changlitang/models/D-ORCA-8B-0210 \
    --model_base /mnt/sh/mmvision/home/changlitang/models/D-ORCA-8B-0210 \
