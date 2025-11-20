import os
import torch

class BaseActivationSaver:
	"""
	Base class for saving activations from different models.
	"""
	def __init__(self, base_save_dir: str, task_id: str, data_split: str, model_name: str, prompt_id: str):
		self.task_id = task_id
		self.data_split = data_split
		self.model_name = model_name
		self.prompt_id = prompt_id
		self.base_save_dir = base_save_dir
		self.current_id = None
		self.current_lang = None

	def set_id(self, new_id):
		self.current_id = new_id

	def set_lang(self, new_lang):
		self.current_lang = new_lang

	def hook_fn(self, module, input, output, layer_id):
		raise NotImplementedError("This method should be overridden by subclasses.")

	def pre_hook_fn(self, module, input, layer_id):
		raise NotImplementedError("This method should be overridden by subclasses.")

	# Check if activations for an instance already exist
	def check_exists(self):
		path_last_token = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
		path_average = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
		check_files = os.listdir(path_last_token) if os.path.exists(path_last_token) else []

		# # Check if each extraction exist
		# post_attn_files = [f for f in check_files if 'postattn' in f]
		# post_mlp_files = [f for f in check_files if 'postmlp' in f]
		# embed_token_file = [f for f in check_files if 'embed_tokens' in f]
		# if len(post_attn_files) == 0 or len(post_mlp_files) == 0 or len(embed_token_file) == 0:
		# 	return False
		
		# Check average directory files
		check_files_avg = os.listdir(path_average) if os.path.exists(path_average) else []

		# If there are any files, return True
		return bool(check_files) and bool(check_files_avg)

	def _save_activation_last_token(self, tensor, layer_id):
		path = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "last_token")
		os.makedirs(path, exist_ok=True)
		save_path = os.path.join(path, f"layer_{layer_id}.pt")
		torch.save(tensor[0, -1, :].detach().cpu(), save_path)

	def _save_activation_average(self, tensor, layer_id):
		path = os.path.join(self.base_save_dir, self.task_id, self.data_split, self.model_name.split('/')[-1], self.prompt_id, self.current_lang, self.current_id, "average")
		os.makedirs(path, exist_ok=True)
		save_path = os.path.join(path, f"layer_{layer_id}.pt")
		torch.save(tensor[0].mean(dim=0).detach().cpu(), save_path)
	
	def _check_set_id_lang(self, layer_id):
		if self.current_id is None:
			print(f"Warning: ID not set for layer {layer_id}")
			return False

		if self.current_lang is None:
			print(f"Warning: Language not set for layer {layer_id}")
			return False
		
		return True
	
class GeneralActivationSaver(BaseActivationSaver): # Handle Gemma3, Qwen, Pythia models (models that activation are returned in form of a tuple)

	def hook_fn(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id) # Unpack tensor from the tuple
			self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn for layer {layer_id}: {e}")
	
	def pre_hook_fn(self, module, input, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return

		try:
			# Extract residual connection after attention (precisely after post-attention layer norm)
			self._save_activation_last_token(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
			self._save_activation_average(tensor=input[0] if isinstance(input, tuple) else input, layer_id=layer_id)
		except Exception as e:
			print(f"Error in pre_hook_fn for layer {layer_id}: {e}")

class CohereDecoderActivationSaver(BaseActivationSaver):
	def __init__(self, base_save_dir: str, task_id: str, data_split: str, model_name: str, prompt_id: str):
		super().__init__(base_save_dir, task_id, data_split, model_name, prompt_id)
		self.initial_residual = None
		self.attn_output = None
	
	def hook_fn_embed_tokens(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		
		try:
			self._save_activation_last_token(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
			self._save_activation_average(tensor=output[0] if isinstance(output, tuple) else output, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn_embed_tokens for layer {layer_id}: {e}")

	def hook_fn_set_initial_residual(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		
		self.initial_residual = input[0] if isinstance(input, tuple) else input
	
	def hook_fn_set_attn_output(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return
		self.attn_output = output[0] if isinstance(output, tuple) else output
	
	def hook_fn_final_output(self, module, input, output, layer_id):
		if self._check_set_id_lang(layer_id) is False:
			return

		# Compute residual post MLP
		if self.initial_residual is None or self.attn_output is None:
			print(f"Warning: Missing stored tensors for layer {layer_id}")
			raise ValueError("Stored tensors are None")
		
		residual_post_mlp = output[0] if isinstance(output, tuple) else output
		residual_post_attn = self.initial_residual + self.attn_output

		try:
			self._save_activation_last_token(tensor=residual_post_attn, layer_id=layer_id.replace('residual-postmlp', 'residual-postattn'))
			self._save_activation_average(tensor=residual_post_attn, layer_id=layer_id.replace('residual-postmlp', 'residual-postattn'))
		except Exception as e:
			print(f"Error in hook_fn_final_output for layer {layer_id.replace('residual-postmlp', 'residual-postattn')}: {e}")
		
		try:
			self._save_activation_last_token(tensor=residual_post_mlp, layer_id=layer_id)
			self._save_activation_average(tensor=residual_post_mlp, layer_id=layer_id)
		except Exception as e:
			print(f"Error in hook_fn_final_output for layer {layer_id}: {e}")

		# Reset stored tensors
		self.initial_residual = None
		self.attn_output = None