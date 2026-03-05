#!/bin/bash
# Evaluation script for Vietnamese ABSA models

# Set MODEL_TYPE to either "task-adaptation" or "supervised"
MODEL_TYPE=${1:-task-adaptation}

TOKENIZER_PATH=VietAI/vit5-base
MODEL_PATH=output/vit5-base-vietnamese-${MODEL_TYPE}
PREDICTION_FILE=${MODEL_PATH}/predict_text_generations.jsonl

echo "Evaluating ${MODEL_TYPE} model..."
echo "Model path: ${MODEL_PATH}"
echo "Prediction file: ${PREDICTION_FILE}"
echo ""

python evaluate.py \
    --tokenizer-path $TOKENIZER_PATH \
    --prediction-path $PREDICTION_FILE

echo ""
echo "Evaluation complete!"
echo "To evaluate the other model type, run:"
echo "  bash scripts/eval_vit5_vietnamese.sh supervised"
echo "  bash scripts/eval_vit5_vietnamese.sh task-adaptation"

