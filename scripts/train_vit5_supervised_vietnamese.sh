#!/bin/bash
# Training script for Vietnamese ABSA using Supervised approach
# Optimized for Kaggle/Colab (no DeepSpeed required)
# Recommended for: larger datasets, maximum in-domain performance

set -x

MODEL_NAME_OR_PATH=VietAI/vit5-base
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset_configs/supervised_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
OUTPUT_DIR=output/vit5-base-vietnamese-supervised

RUN_NAME=vit5-vietnamese-supervised

# Standard training without DeepSpeed
python src/run.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_average_f1" \
    --greater_is_better True \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-05 \
    --num_train_epochs 15 \
    --run_name $RUN_NAME \
    --max_source_length 512 \
    --max_target_length 512 \
    --generation_max_length 512 \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --seed 42 \
    --fp16 \
    --report_to none