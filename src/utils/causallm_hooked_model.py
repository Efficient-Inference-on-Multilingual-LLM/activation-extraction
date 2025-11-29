from .activation_saver import BaseActivationSaver, CohereDecoderActivationSaver
from transformers import AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

load_dotenv() 

class BaseHookedModel:
	"""
	Base class for hooking into different models.
	"""
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		device = "cpu"
		model_dtype = torch.float16
		if torch.cuda.is_available():
			device = "cuda"
			compute_capability = torch.cuda.get_device_capability()[0]

			# Use bfloat16 if supported
			if compute_capability >= 8:
				model_dtype = torch.bfloat16
		
		self.model_name = model_name
		self.saver = saver
		self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, device_map=device, cache_dir=os.getenv("HF_CACHE_DIR"))
		self.model.eval()
	
	def _setup_hooks(self):
		raise NotImplementedError("This method should be overridden by subclasses.")
	
	def set_saver_id(self, new_id: int):
		self.saver.set_id(new_id)

	def set_saver_lang(self, new_lang: str):
		self.saver.set_lang(new_lang)
		
	def generate(self, inputs):
		with torch.no_grad():
			outputs = self.model.generate(
				**inputs,
				max_new_tokens=1,
			)
		return outputs

	# Clear hooks for debugging purposes
	def clear_hooks(self):
		if 'bloom' in self.model_name:
			for i, layer in enumerate(self.model.transformer.h):
				layer._forward_hooks.clear()
		else:
			self.model.model.embed_tokens._forward_hooks.clear()
			self.model.model.norm._forward_hooks.clear()
			for i, layer in enumerate(self.model.model.layers):
				layer._forward_hooks.clear()
				layer._forward_pre_hooks.clear()
				
	
class Gemma3MultimodalHookedModel(BaseHookedModel): # For gemma-3 >=4b
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		self.model.model.language_model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.language_model.layers):

			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention layer norm pre-hook (residual post attention)
			layer.pre_feedforward_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

class PythiaHookedModel(BaseHookedModel): # For pythia models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.gpt_neox.embed_in.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.gpt_neox.layers):

			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

class CohereDecoderHookedModel(BaseHookedModel): # For cohere decoder models
	def __init__(self, model_name: str, saver: CohereDecoderActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn_embed_tokens(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Init residual hook
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-init_{i}": self.saver.hook_fn_set_initial_residual(module, input, output, layer_id))

			# Post-attention layer norm pre-hook (residual post attention)
			layer.self_attn.register_forward_hook(lambda module, input, output, layer_id=f"residual-postattn_{i}": self.saver.hook_fn_set_attn_output(module, input, output, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn_final_output(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn_input_layernorm(module, input, output, layer_id))

class Qwen3HookedModel(BaseHookedModel): # For Qwen models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))

class LlamaHookedModel(BaseHookedModel): # For Llama 3 models
	def __init__(self, model_name: str, saver: BaseActivationSaver):
		super().__init__(model_name, saver)
		self._setup_hooks()

	def _setup_hooks(self):
		# Embedding layer
		self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))

		# Decoder layers
		for i, layer in enumerate(self.model.model.layers):

			# Post-attention layer norm pre-hook (residual post attention)
			layer.post_attention_layernorm.register_forward_pre_hook(lambda module, input, layer_id=f"residual-postattn_{i}": self.saver.pre_hook_fn(module, input, layer_id))
			
			# Final output of decoder layer hook
			layer.register_forward_hook(lambda module, input, output, layer_id=f'residual-postmlp_{i}': self.saver.hook_fn(module, input, output, layer_id))

			# Pre-attention layer norm hook (residual pre attention)
			layer.input_layernorm.register_forward_hook(lambda module, input, output, layer_id=f"residual-preattn_{i}": self.saver.hook_fn(module, input, output, layer_id))