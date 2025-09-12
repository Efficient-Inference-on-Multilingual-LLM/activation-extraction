from .activation_saver import ActivationSaver
from transformers import AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

load_dotenv() 

class HookedModel:
    def __init__(self, model_name: str, saver: ActivationSaver):

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
        self._setup_hooks()

    def _setup_hooks(self):
        if 'bloom' in self.model_name:
            for i, layer in enumerate(self.model.transformer.h):
                layer.register_forward_hook(lambda module, input, output, layer_id=i: self.saver.hook_fn(module, input, output, layer_id))
        else:
            self.model.model.embed_tokens.register_forward_hook(lambda module, input, output, layer_id="embed_tokens": self.saver.hook_fn(module, input, output, layer_id))
            self.model.model.norm.register_forward_hook(lambda module, input, output, layer_id="norm": self.saver.hook_fn(module, input, output, layer_id))
            for i, layer in enumerate(self.model.model.layers):
                layer.register_forward_hook(lambda module, input, output, layer_id=i: self.saver.hook_fn(module, input, output, layer_id))
            
            # TODO: Figure out how to extract components of the model that have been applied with RoPE
            # self.model.model.rotary_emb.register_forward_hook(lambda module, input, output, layer_id="rotary_emb": self.saver.hook_fn(module, input, output, layer_id))
        

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

            # self.model.model.rotary_emb._forward_hooks.clear()