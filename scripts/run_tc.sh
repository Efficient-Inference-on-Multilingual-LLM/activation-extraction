#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

echo "Running inference at $(date)"
python run_activation_tc.py \
    --model_name "Qwen/Qwen3-4B" \
    --prompt_path "./prompts/topic_classification/prompt_en.txt" \
    --output_dir "./outputs" \
    --languages ind_Latn eng_Latn \
    --is_base_model
echo "--------------------------------------------------------"
echo "========================================================"
echo "Completed at: $(date)"
echo "========================================================"
