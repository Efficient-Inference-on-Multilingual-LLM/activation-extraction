import argparse
from tqdm import tqdm
from datasets import load_dataset
from ..utils.const import LANGCODE2LANGNAME
from ..utils.hooked_model import HookedModel
from ..utils.activation_saver import ActivationSaver
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main(args):
    # Load Model
    print(f'Load model: {args.model_name}')
    saver = ActivationSaver(args.output_dir, task_id='topic_classification', model_name=args.model_name, prompt_id= "prompted" if args.is_prompted == "True" else "raw")
    hooked_model = HookedModel(args.model_name, saver=saver)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Feed Forward
    for lang in args.languages:
        # Load Dataset
        datasets_per_lang = {}
        datasets_per_lang[lang] = load_dataset("Davlan/sib200", lang, split="test")

        # Load Prompt Template
        if args.is_prompted == "True": 
            with open(f'./prompts/topic_classification/{lang}.txt') as f:
                prompt_template = f.read()
        
        # Iterate Through Each Instance
        for instance in tqdm(datasets_per_lang[lang], desc=f"Processing activation for topic classification task ({lang})"):
            hooked_model.set_saver_id(str(instance['index_id']))
            hooked_model.set_saver_lang(lang)

            # Build Prompt Based on Template
            if args.is_prompted == "True": 
                prompt = prompt_template.replace("{text}", instance['text'])
            else:
                prompt = instance['text']

            # Inference
            if args.is_base_model or 'bloom' in args.model_name:
                text = prompt
            else:
                messages = [{'role': 'user', 'content': prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False 
                )
            
            inputs = tokenizer([text], return_tensors="pt").to(hooked_model.model.device)
            _ = hooked_model.generate(inputs)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Extract activation for topic classification task")
	parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
	parser.add_argument("--is_prompted", type=str, default="True", help="Is Prompted")
	parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
	parser.add_argument('--languages', type=str, nargs='+', default=['fra_Latn', 'eng_Latn', 'ind_Latn'], help='List of languages')
	parser.add_argument('--is_base_model', action='store_true', help='Whether the model is a base model or a instruct model')

	args = parser.parse_args()
	main(args)
