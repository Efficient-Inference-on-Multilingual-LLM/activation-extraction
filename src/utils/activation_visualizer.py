from typing import List, Dict, Tuple, Literal, Any
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score
from tqdm import tqdm

class ActivationVisualizer:
    def __init__(
            self, 
            models: List[Dict], 
            languages: List[str],
            data: Any = None,
        ):
        self.models = models
        self.languages = languages
        assert data is not None, "Data must be provided"
        self.data = data
        self.topics = self.data[self.languages[0]]['category'].unique().tolist() if self.data is not None else []

    def _create_color_map(self, plot_by: Literal["topic", "language"]) -> Dict[str, Tuple]:
        cmap = plt.get_cmap('tab20')
        if plot_by == "topic":  
            return {topic: cmap(i) for i, topic in enumerate(self.topics)}
        elif plot_by == "language":
            return {language: cmap(i) for i, language in enumerate(self.languages)}

    def generate_plots_classification(
            self,
            save_path: str = "results",
            ext: Literal["png", "jpg", "jpeg", "pdf"] = "pdf",
            activation_path: str = "./activations/extracted",
            input_mode: Literal["raw", "prompted"] = "raw",
            extraction_mode: Literal["last_token", "average", "first_token"] = "average",
            plot_by: Literal["topic", "language"] = "language",
        ):
        self.color_map = self._create_color_map(plot_by=plot_by)
        for model_id in range (len(self.models)):
            fig, axes = plt.subplots(int(np.ceil((self.models[model_id]['num_layers']+1)/8)), 8, figsize=(32, int((self.models[model_id]['num_layers']+1)/2)))
            axes = axes.flatten()

            for layer in tqdm(range(-1, self.models[model_id]['num_layers']), desc = f"Processing Model {self.models[model_id]['name']} Layers"):
                label_language = []
                latent = []
                
                for current_language in self.languages:
                    if activation_path is None:
                        raise ValueError("activation_path must be provided")
                    base_path = os.path.join(activation_path, self.models[model_id]['name'], input_mode, current_language, current_language)
                    for text_id in os.listdir(base_path):
                        text_path = os.path.join(base_path, text_id)
                        if not os.path.isdir(text_path):
                            continue
                        path = os.path.join(text_path, extraction_mode, f"layer_{"embed_tokens" if layer == -1 else layer}.pt")
                        if not os.path.exists(path):
                            print(f"Warning: File {path} does not exist, skipping...")
                            continue
                        try:
                            activation_values = torch.load(path)
                        except EOFError:
                            print(f"Error loading {path}, skipping...")
                            continue
                        latent.append(activation_values.to(torch.float32).numpy())
                        if plot_by == "topic":
                            current_data = self.data[current_language]
                            matching_row = current_data[current_data['index_id'].astype(str) == text_id]
                            if matching_row.empty:
                                raise ValueError(f"Warning: No matching data found for text_id {text_id} in language {current_language}")
                            label_language.append(matching_row['category'].values[0])
                        elif plot_by == "language":
                            label_language.append(current_language)

                latent = np.array(latent)
                score = silhouette_score(latent, label_language)
                tsne = TSNE(n_components=2, random_state=42)
                latent_2d = tsne.fit_transform(latent)

                ax = axes[layer+1]
                if plot_by == "topic":
                    for topic in self.topics:
                        indices = [i for i, lang in enumerate(label_language) if lang == topic]
                        ax.scatter(latent_2d[indices, 0], latent_2d[indices, 1], label=topic, color=self.color_map[topic], alpha=0.6, s=10)
                elif plot_by == "language":
                    for language in self.languages:
                        indices = [i for i, lang in enumerate(label_language) if lang == language]
                        ax.scatter(latent_2d[indices, 0], latent_2d[indices, 1], label=language, color=self.color_map[language], alpha=0.6, s=10)
                
                ax.set_title(f"Layer {'embed_tokens' if layer == -1 else layer} \nSilhouette Score: {score:.2f}")
            
            if plot_by == "topic":
                legend = [Line2D([0], [0], marker='o', color='w', label=topic, markerfacecolor=self.color_map[topic], markersize=8, alpha=0.6) for topic in self.topics]
            elif plot_by == "language":
                legend = [Line2D([0], [0], marker='o', color='w', label=lang, markerfacecolor=self.color_map[lang], markersize=8, alpha=0.6) for lang in self.languages]

            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            ncol = len(self.topics) if plot_by == "topic" else len(self.languages)
            fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=ncol, title="Languages" if plot_by == "language" else "Topics")
            os.makedirs(f"{save_path}", exist_ok=True)
            plt.savefig(f"{save_path}/{self.models[model_id]['name']}_{plot_by}_{input_mode}_{extraction_mode}.{ext}", bbox_inches='tight')
            plt.show()

    def generate_silhoutte_score(self):
        num_models = len(self.models)
        fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 6))

        language_scores = []
        topic_scores = []
        text_id_scores = []

        # for layer in range (-1, )

    # TODO: Machine Translation plotting    
    def generate_plots_machine_translation(
            self, 
        ):
        for model_id in range (len(self.models)):
            for language in self.languages:
                fig, axes = plt.subplots(self.models[model_id]['row'], self.models[model_id]['col'], figsize=self.models[model_id]['figsize'])

                for layer in range (self.models[model_id]['num_layers']):
                    label_language = []
                    latent = []

                    target_language = language
                    for source_language in self.languages:
                        if source_language != target_language:
                            label_language.append(source_language)
                            latent.append(self.models[model_id]['data'][source_language][target_language][layer])
