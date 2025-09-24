import pickle
import os
from tqdm import tqdm
from typing import Dict, Literal, List, Any
import torch
from sklearn.metrics import silhouette_score
import numpy as np
import traceback
from ..utils.const import LANGCODE2LANGNAME, LANGNAME2LANGCODE, EXPERIMENT8_LANGUAGES
from datasets import load_dataset
from dotenv import load_dotenv
import cupy as cp # Use CuPy for GPU arrays

# Try to import cuML's TSNE, fallback to sklearn's TSNE if not available
try:
    from cuml.manifold import TSNE as cuTSNE
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
from sklearn.manifold import TSNE as skTSNE

load_dotenv()  # Load environment variables from .env file

def extract_tsne_data(
        models: List[Dict],
        languages: List[str],
        data: Dict[str, Any], # SIB-200 dataset
        save_path: str = "results",
        activation_path: str = "./outputs/topic_classification",
        input_mode: Literal["raw", "prompted", "prompt_eng_Latn", "prompt_ind_Latn"] = "raw",
        extraction_mode: Literal["last_token", "average", "first_token"] = "average",
        plot_by: Literal["topic", "language"] = "language",
        device: Literal["cpu", "cuda"] = "cpu",
    ):
    """
    Extract t-SNE latent representations and labels, save to pickle files.
    
    Returns:
        Dict: Dictionary containing extracted data for each model and layer
    """
    extracted_data = {}
    
    for model in models:
        model_name = model['name']
        num_layers = model['num_layers']
        extracted_data = {}
        
        # Iterate Through Each Layer (from embed_tokens to last layer)
        for layer in tqdm(range(-1, num_layers), desc=f"Processing Model {model_name} Layers"):
            # Initialize lists to hold labels and latent representations
            label_language = []
            latent = []
            
            # Loop through each language
            for current_language in languages:
                if activation_path is None:
                    raise ValueError("activation_path must be provided")
                
                # Construct base path for the current model, input mode, and language
                base_path = os.path.join(activation_path, model_name.split('/')[-1], input_mode, current_language)

                # Iterate through each text_id directory
                for text_id in os.listdir(base_path):
                    text_path = os.path.join(base_path, text_id)
                    
                    if not os.path.isdir(text_path):
                        continue
                    
                    path = os.path.join(text_path, extraction_mode, f"layer_{'embed_tokens' if layer == -1 else layer}.pt")
                    if not os.path.exists(path):
                        print(f"Warning: File {path} does not exist, skipping...")
                        continue
                    
                    # Load activation values
                    try:
                        activation_values = torch.load(path)
                    except EOFError:
                        print(f"Error loading {path}, skipping...")
                        continue
                    latent.append(activation_values.to(torch.float32).numpy())
                    
                    # Determine label based on plot_by parameter
                    if plot_by == "topic":
                        current_data = data[current_language]
                        matching_row = current_data[current_data['index_id'].astype(str) == text_id]
                        if matching_row.empty:
                            raise ValueError(f"Warning: No matching data found for text_id {text_id} in language {current_language}")
                        label_language.append(matching_row['category'].values[0])
                    elif plot_by == "language":
                        label_language.append(LANGCODE2LANGNAME[current_language])
            
            # Convert lists to numpy arrays
            latent = np.array(latent)

            # Compute silhouette score
            score = silhouette_score(latent, label_language)

            # Perform t-SNE dimensionality reduction, using cuML if on GPU
            if device == "cuda" and CUML_AVAILABLE:
                latent = cp.asarray(latent)
                tsne = cuTSNE(n_components=2, random_state=42)
            else:
                tsne = skTSNE(n_components=2, random_state=42)

            try:
                latent_2d = tsne.fit_transform(latent)
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                continue
            
            # Store results in the dictionary
            layer_key = 'embed_tokens' if layer == -1 else f'layer_{layer}'
            extracted_data[layer_key] = {
                'latent_2d': cp.asnumpy(latent_2d) if device == "cuda" else latent_2d,
                'labels': label_language,
                'silhouette_score': score,
            }
    
        # Save to pickle file
        pickle_filename = os.path.join(save_path, model_name.split('/')[-1], input_mode, extraction_mode, f"{plot_by}_tsne.pkl")
        os.makedirs(os.path.dirname(pickle_filename), exist_ok=True)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(extracted_data, f)
        
        print(f"t-SNE data saved to {pickle_filename}")

    return extracted_data

if __name__ == "__main__":
    languages = EXPERIMENT8_LANGUAGES
    data = {}
    for language in tqdm(languages, desc="Loading SIB-200 Dataset"):
        data[language] = load_dataset('Davlan/sib200', LANGNAME2LANGCODE[language], split="test", cache_dir=os.getenv("HF_CACHE_DIR"))
    
    langcodes = [LANGNAME2LANGCODE[lang] for lang in languages]
    _ = extract_tsne_data(
        models=[
            {'name': 'google/gemma-3-4b-it', 'num_layers': 34},
            {'name': 'google/gemma-3-1b-it', 'num_layers': 26},
            {'name': 'google/gemma-3-270m-it', 'num_layers': 18},
        ],
        languages=langcodes,
        data=data,
        save_path="./outputs_tsne/topic_classification",
        activation_path="./outputs/topic_classification",
        input_mode="raw", # "raw" or "prompted" or "prompt_eng_Latn" or "prompt_ind_Latn"
        extraction_mode="last_token", # "last_token" or "average" or "first_token"
        plot_by="language", # "topic" or "language"
        device="cuda" if torch.cuda.is_available() else "cpu",
    )