#!/bin/bash

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

# Model and experiment version
MODEL_NAME="Qwen/Qwen3-4B"
    
echo "Running inference at $(date)"
python run_activation_mt.py \
    --model_name $MODEL_NAME \
    --prompt_path "./prompts/mt/prompt_en.txt" \
    --output_dir "./outputs/mt-testing" \
    --target_langs fra_Latn jav_Latn sun_Latn tur_Latn cym_Latn \
    --source_langs ind_Latn eng_Latn \
    # --source_langs fra_Latn jav_Latn sun_Latn tur_Latn cym_Latn \
    # --target_langs ind_Latn eng_Latn \
    # --is_base_model
echo "--------------------------------------------------------"
echo "========================================================"
echo "Completed at: $(date)"
echo "========================================================"
