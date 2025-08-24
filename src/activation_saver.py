import os
import torch

class ActivationSaver:
    def __init__(self, base_save_dir: str, task_id: str, model_name: str, prompt_id: str):
        self.task_id = task_id
        self.model_name = model_name
        self.prompt_id = prompt_id
        self.base_save_dir = base_save_dir
        self.current_id = None
        self.current_lang = "en"

    def set_id(self, new_id):
        self.current_id = new_id

    def set_lang(self, new_lang):
        self.current_lang = new_lang

    def hook_fn(self, module, input, output, layer_id):
        if self.current_id is None:
            print(f"Warning: ID not set for layer {layer_id}")
            return

        if self.current_lang is None:
            print(f"Warning: Language not set for layer {layer_id}")
            return

        path = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id)
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"layer_{layer_id}.pt")

        # TODO: Adjust the indexing of the activations based on the type of the models
        if any(list(map(lambda x: x in self.model_name.lower(), ['qwen']))):
            torch.save(output[0, -1, :].detach().cpu(), save_path) 
        else:
            torch.save(output[0][0, -1, :].detach().cpu(), save_path) 