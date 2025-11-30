import os
import sys
import pandas as pd
import numpy as np
from src.utils.const import LANGCODE2LANGNAME, LANGNAME2LANGCODE, MODEL2HIDDEN_SIZE, MODEL2NUM_LAYERS, EXP2_CONFIG, EXP3_CONFIG, EXP4_CONFIG
import glob
import torch
from tqdm import tqdm
import cudf
import cupy as cp
from cuml.cluster import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from sklearn.datasets import make_blobs
from typing import List, Literal

def calculate_silhouette_score(
        output_dir: str, 
        activation_dir: str,
        task: str,
        data_split: str,
        model_name: str,
        extraction_mode: str,
        token_position: Literal['last_token', 'average'],
        residual_locations: List[Literal["residual-postattn", "residual-postmlp"]],
        languages: List[str]):
    
    num_layers = MODEL2NUM_LAYERS[model_name]
    text_ids = glob.glob(f'{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/eng_Latn/*')
    text_ids = [text_id.split('/')[-1].split('.')[0] for text_id in text_ids]

    print(f"Text IDs: {len(text_ids)}, Num layers: {num_layers}, Number of languages: {len(languages)}, Hidden size: {MODEL2HIDDEN_SIZE[model_name]}")
    
    # Pre-build all file paths once to avoid repeated glob operations
    file_paths = {}
    for lang in tqdm(languages, desc='Building file paths', total=len(languages)):
        file_paths[lang] = {
            'embed': sorted(glob.glob(f'{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/{lang}/*/{token_position}/layer_embed_tokens.pt')),
        }
        for residual_location in residual_locations:
            for layer_id in range(num_layers):
                key = f'{residual_location}_{layer_id}'
                file_paths[lang][key] = sorted(glob.glob(f'{activation_dir}/{task}/{data_split}/{model_name}/{extraction_mode}/{lang}/*/{token_position}/layer_{residual_location}_{layer_id}.pt'))
    
    # Load activations per residual location
    for residual_location in residual_locations:
        print(f'Calculating silhouette scores for residual location: {residual_location}')

        # Initialize labels once (use int32 instead of long to save memory)
        labels = torch.zeros((len(text_ids) * len(languages),), dtype=torch.int32, device='cuda')
        
        # Vectorized label assignment
        labels = torch.arange(len(languages), device='cuda').repeat_interleave(len(text_ids))
        print(f'Labels tensor shape: {labels.shape}')

        silhouette_score_matrix = torch.zeros((num_layers + 1, len(languages), len(languages)), device='cuda')
        
        for layer_id in tqdm(range(-1, num_layers), desc='Processing layers'):
            # Preallocate activation tensor
            activation_current_layer = torch.empty((len(text_ids) * len(languages), MODEL2HIDDEN_SIZE[model_name]), 
                                                   dtype=torch.float32, device='cuda')
            
            # Load activations more efficiently using batch loading
            for lang_idx, lang in enumerate(languages):
                if layer_id == -1:
                    paths = file_paths[lang]['embed']
                else:
                    key = f'{residual_location}_{layer_id}'
                    paths = file_paths[lang][key]
                
                if len(paths) != len(text_ids):
                    print(f"Warning: Expected {len(text_ids)} files for language '{lang}' at layer {layer_id}, but found {len(paths)} files.")
                    break
                
                # Batch load activations for this language
                for text_idx, path in enumerate(paths):
                    flat_idx = text_idx * len(languages) + lang_idx
                    activation = torch.load(path, map_location='cuda')  # Load directly to GPU
                    activation_current_layer[flat_idx] = activation
            
            # Pre-compute language pair masks once
            lang_pair_masks = {}
            for lang_idx1 in range(len(languages)):
                for lang_idx2 in range(lang_idx1 + 1, len(languages)):  # Only upper triangle
                    mask = (labels == lang_idx1) | (labels == lang_idx2)
                    lang_pair_masks[(lang_idx1, lang_idx2)] = mask
            
            # Calculate pairwise silhouette scores (only upper triangle, then mirror)
            for lang_idx1 in range(len(languages)):
                for lang_idx2 in range(lang_idx1 + 1, len(languages)):
                    mask = lang_pair_masks[(lang_idx1, lang_idx2)]
                    activations_pair = activation_current_layer[mask]
                    labels_pair = labels[mask]
                    
                    score = cython_silhouette_score(activations_pair, labels_pair)
                    
                    # Store in both positions (symmetric matrix)
                    silhouette_score_matrix[layer_id + 1, lang_idx1, lang_idx2] = score
                    silhouette_score_matrix[layer_id + 1, lang_idx2, lang_idx1] = score
            
            # Free memory after each layer
            del activation_current_layer
            torch.cuda.empty_cache()

        # Store silhouette score matrix
        output_path = os.path.join(output_dir, task, data_split, model_name, extraction_mode, token_position, residual_location)
        os.makedirs(output_path, exist_ok=True)
        torch.save(silhouette_score_matrix, os.path.join(output_path, 'silhouette_score_matrix.pt'))
        
        # Clean up
        del labels, silhouette_score_matrix
        torch.cuda.empty_cache()


if __name__ == "__main__":
    languages = []
    for family, langs in EXP4_CONFIG['languages'].items():
        languages.extend(langs)

    languages = [LANGNAME2LANGCODE[lang] for lang in languages]
    output_dir = f'outputs_silhouette/{EXP4_CONFIG["exp_id"]}'
    model_names = [
        # 'gemma-3-4b-it',
        # 'Llama-3.1-8B-Instruct',
        # 'aya-expanse-8b',
        # 'Qwen3-8B',
        # 'Qwen3-14B',
        'gemma-3-12b-it',
        # 'pythia-6.9b-deduped',
        # 'aya-101'
    ]
    for model_name in tqdm(model_names, desc="Calculating silhouette scores for models"):
        calculate_silhouette_score(
            output_dir=output_dir,
            activation_dir=f'outputs_flores_plus',
            task='next_token',
            data_split='dev',
            model_name=model_name,
            extraction_mode='raw',
            token_position='last_token',
            residual_locations=['residual-postmlp', 'residual-postattn', 'residual-preattn'],
            languages=languages
        )
    