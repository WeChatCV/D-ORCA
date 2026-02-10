#!/bin/bash

cd "$(cd $(dirname $0); pwd)"

bash test.sh \
    --interval 0.5 \
    --run_name debug2 \
    --dataset /mnt/sh/mmvision/home/changlitang/test/tmp.json \
    --max_frames 128 \
    --model /mnt/sh/mmvision/home/changlitang/models/D-ORCA-8B-0210 \
    --model_base /mnt/sh/mmvision/home/changlitang/models/D-ORCA-8B-0210 \
    --gpu_num 1 --max_new_tokens 3072
