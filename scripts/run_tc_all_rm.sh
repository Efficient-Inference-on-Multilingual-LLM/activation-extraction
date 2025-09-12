#!/bin/bash

# Array of model names
model_names=(
#    "meta-llama/Meta-Llama-3-8B"
    "google/gemma-2-9b-it"
#    "google/gemma-2-9b"
#    "aisingapore/Gemma-SEA-LION-v3-9B"
#    "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-base"
#    "aisingapore/Llama-SEA-LION-v3-8B"
#    "GoToCompany/llama3-8b-cpt-sahabatai-v1-base"
#    "Qwen/Qwen2.5-7B"
#    "sail/Sailor2-8B"
    "sail/Sailor2-8B-Chat"
    "CohereLabs/aya-expanse-8b"
)

# Specifiy cuda device if needed
export CUDA_VISIBLE_DEVICES=0

# Loop through each model
for model_name in "${model_names[@]}"; do
    echo "Running inference for model: $model_name at $(date)"
    python3 -m src.main.run_activation_tc \
        --model_name "$model_name" \
        --prompt_lang "eng_Latn" \
        --output_dir "./outputs" \
        --languages eng_Latn fra_Latn ind_Latn kor_Hang jpn_Jpan sun_Latn jav_Latn \
#        --is_base_model \
#	--sample_size 100
    echo "--------------------------------------------------------"
    rm -rf ~/.cache/huggingface/hub/*
done

echo "========================================================"
echo "All models completed at: $(date)"
echo "========================================================"
