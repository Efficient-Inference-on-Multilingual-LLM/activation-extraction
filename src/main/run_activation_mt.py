import argparse
from tqdm import tqdm
from datasets import load_dataset
from ..utils.const import LANGCODE2LANGNAME
from ..utils.hooked_model import HookedModel
from ..utils.activation_saver import ActivationSaver
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main(args):
	with open(args.prompt_path) as f:
		initial_prompt = f.read()
	# Get the file name and remove the extension
	prompt_id = os.path.basename(args.prompt_path).split('.')[0]

	# Load datasets
	datasets_per_lang = {}
	for source_langs in args.source_langs:
		datasets_per_lang[source_langs] = load_dataset("openlanguagedata/flores_plus", source_langs, split="devtest")

	# Load hooked model
	print(f'Load model: {args.model_name}')
	saver = ActivationSaver(args.output_dir, task_id='machine_translation', model_name=args.model_name, prompt_id=prompt_id)
	hooked_model = HookedModel(args.model_name, saver=saver)  # Initialize with a hook fn saver
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)

	for target_lang in args.target_langs:
		for source_lang in args.source_langs:
			for instance in tqdm(datasets_per_lang[source_lang], desc=f"Processing activation for translation task: {source_lang} to {target_lang}"):

				# Setup activation save location
				hooked_model.set_saver_id(str(instance['id']))
				hooked_model.set_saver_lang(f"{source_lang}-{target_lang}")

				# Setup prompt
				prompt = initial_prompt.replace("{text}", instance['text'])
				prompt = prompt.replace("{source_lang}", LANGCODE2LANGNAME[source_lang])
				prompt = prompt.replace("{target_lang}", LANGCODE2LANGNAME[target_lang])

				# Add template for instruct models
				if args.is_base_model or 'bloom' in args.model_name:
					text = prompt
				else:
					messages = [
						{'role': 'system', 'content': ''},
						{'role': 'user', 'content': prompt}
					]
					if 'meta-llama' in args.model_name.lower():
						user_prompt = messages[-1]['content']
						text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
					else:
						text = tokenizer.apply_chat_template(
							messages,
							tokenize=False,
							add_generation_prompt=True,
							enable_thinking=False 
						)
				
				# Tokenize inputs
				inputs = tokenizer([text], return_tensors="pt").to(hooked_model.model.device)

				# Run forward prop
				_ = hooked_model.generate(inputs)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Extract activation for Machine Translation task")
	parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
	parser.add_argument("--prompt_path", type=str, default="./prompts/machine_translation/prompt_en.txt", help="Path to the prompt file")
	parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
	parser.add_argument('--source_langs', type=str, nargs='+', default=['fra_Latn', 'jav_Latn', 'sun_Latn', 'tur_Latn', 'cym_Latn'], help='List of source languages')
	parser.add_argument("--target_langs", type=str, nargs='+', default=['ind_Latn', 'eng_Latn'], help="List of target languages")
	parser.add_argument('--is_base_model', action='store_true', help='Whether the model is a base model or a instruct model')

	args = parser.parse_args()
	main(args)
