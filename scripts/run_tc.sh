#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

# Model and experiment version
MODEL_NAME="Qwen/Qwen3-4B"
    
echo "Running inference at $(date)"
python run_activation_tc.py \
    --model_name $MODEL_NAME \
    --prompt_path "./prompts/tc/prompt_en.txt" \
    --output_dir "./outputs/tc-testing" \
    --languages ind_Latn eng_Latn \
    --is_base_model
echo "--------------------------------------------------------"
echo "========================================================"
echo "Completed at: $(date)"
echo "========================================================"
