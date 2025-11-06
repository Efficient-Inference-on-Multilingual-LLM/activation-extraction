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

		if any(list(map(lambda x: x in self.model_name.lower(), ['gemma']))) and layer_id not in ['embed_tokens', 'norm']:
			path = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
			os.makedirs(path, exist_ok=True)
			save_path = os.path.join(path, f"layer_{layer_id}.pt")
			torch.save(output[0][0, -1, :].detach().cpu(), save_path) 

			path = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
			os.makedirs(path, exist_ok=True)
			save_path = os.path.join(path, f"layer_{layer_id}.pt")
			torch.save(output[0][0].mean(dim=0).detach().cpu(), save_path) 

		# Extraction for any other layers/models
		else:
			path = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
			os.makedirs(path, exist_ok=True)
			save_path = os.path.join(path, f"layer_{layer_id}.pt")
			torch.save(output[0, -1, :].detach().cpu(), save_path) 

			path = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
			os.makedirs(path, exist_ok=True)
			save_path = os.path.join(path, f"layer_{layer_id}.pt")
			torch.save(output[0].mean(dim=0).detach().cpu(), save_path)
	
	def pre_hook_fn(self, module, input, layer_id):
		if self.current_id is None:
			print(f"Warning: ID not set for layer {layer_id}")
			return

		if self.current_lang is None:
			print(f"Warning: Language not set for layer {layer_id}")
			return

		# Extract residual connection after attention (precisely after post-attention layer norm)
		save_path = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
		os.makedirs(save_path, exist_ok=True)
		save_path = os.path.join(save_path, f"layer_{layer_id}.pt")
		torch.save(input[0][0, -1, :].detach().cpu(), save_path)

	# Check if activations for an instance already exist
	def check_exists(self):
		path_last_token = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
		path_average = os.path.join(self.base_save_dir, self.task_id, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
		check_files = os.listdir(path_last_token) if os.path.exists(path_last_token) else []

		# Check if post-attention norm files exist as well
		post_attn_files = [f for f in check_files if 'postattn-norm' in f]
		if len(post_attn_files) == 0:
			return False
		
		# Check average directory files
		check_files += os.listdir(path_average) if os.path.exists(path_average) else []

		# If there are any files, return True
		return bool(check_files)