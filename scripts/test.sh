#!/bin/bash
cd "$(cd $(dirname $0); pwd)/.."
echo "All parameters: $@"

export HF_HOME="/mnt/sh/mmvision/home/changlitang/huggingface"

DATASET=1
MODEL=1
MODEL_BASE=1
LR=2e-5
BS=1
ACCUM_STEPS=1
RUN_NAME="debug"
DEEPSPEED=0
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
EPOCH=1

MAX_PIXELS=176400
MIN_PIXELS=784

SAVE_STEPS=1000

MIN_FRAMES=1
MAX_FRAMES=128
INTERVAL=0.2

USE_LORA=False
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
LORA_CKPT=No
NUM_SAMPLE=1
DO_SAMPLE=False

GPU_NUM=8
MAX_NEW_TOKENS=1024
NUM_GENERATIONS=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --train_llm) TRAIN_LLM=True ;;
        --train_proj) TRAIN_PROJ=True ;;
        --train_enc) TRAIN_ENC=True ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --epoch) EPOCH="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --use_lora) USE_LORA=True ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --do_sample) DO_SAMPLE=True ;;
        --num_sample) NUM_SAMPLE="$2"; shift ;;
        --gpu_num) GPU_NUM="$2"; shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --num_generations) NUM_GENERATIONS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

rm -rf tmp_dataset
mkdir -p output
mkdir -p tmp_dataset
cp $DATASET tmp_dataset
DATASET=tmp_dataset/$(basename $DATASET)

python3 scripts/split_data.py $DATASET $(($WORLD_SIZE*$GPU_NUM)) tmp_dataset # 8

for i in $(seq 0 $(($GPU_NUM - 1))); do
    CUDA_VISIBLE_DEVICES=$i torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=0.0.0.0 --master_port=$((14801 + i + 8 * RANK)) \
        qwenvl/train/eval_dorca.py \
            --model_base $MODEL_BASE \
            --run_test True \
            --pred_rank $((8 * $RANK + i)) \
            --deepspeed ./scripts/zero${DEEPSPEED}.json \
            --model_name_or_path "$MODEL" \
            --dataset_use tmp_dataset/$((8 * $RANK + i)).json \
            --tune_mm_vision $TRAIN_ENC \
            --tune_mm_mlp $TRAIN_PROJ \
            --tune_mm_llm $TRAIN_LLM \
            --bf16 \
            --output_dir output/test \
            --num_train_epochs $EPOCH \
            --per_device_train_batch_size $BS \
            --gradient_accumulation_steps $ACCUM_STEPS \
            --max_pixels $MAX_PIXELS \
            --min_pixels $MIN_PIXELS \
            --eval_strategy "no" \
            --save_strategy "steps" \
            --save_steps $SAVE_STEPS \
            --save_total_limit 5 \
            --learning_rate $LR \
            --weight_decay 0 \
            --warmup_ratio 0.03 \
            --max_grad_norm 1 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --model_max_length 131072 \
            --gradient_checkpointing True \
            --dataloader_num_workers 8 \
            --run_name $RUN_NAME \
            --video_min_frames $MIN_FRAMES \
            --video_max_frames $MAX_FRAMES \
            --base_interval $INTERVAL \
            --use_lora $USE_LORA \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --lora_ckpt $LORA_CKPT \
            --num_sample $NUM_SAMPLE \
            --do_sample $DO_SAMPLE \
            --report_to tensorboard \
            --max_new_tokens $MAX_NEW_TOKENS \
            --num_generations $NUM_GENERATIONS &
done

wait
