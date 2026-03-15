set -x

port=$(shuf -i25000-30000 -n1)

MODEL_NAME_OR_PATH=VietAI/vit5-base
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
OUTPUT_DIR=output/vit5-base-task-adaptation
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero0_t5.json

RUN_NAME=vit5-base-experiment

python src/run.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --save_only_model True\
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_average_f1" \
    --greater_is_better True \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-05 \
    --num_train_epochs 50 \
    --run_name $RUN_NAME \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --report_to "tensorboard" \
    --seed 1234
