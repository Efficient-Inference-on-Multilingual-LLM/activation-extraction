import argparse
from tqdm import tqdm
from datasets import load_dataset
from src.const import LANGCODE2LANGNAME
from src.hooked_model import HookedModel
from src.activation_saver import ActivationSaver
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main(args):
    with open(args.prompt_path) as f:
        initial_prompt = f.read()
    # Get the file name and remove the extension
    prompt_id = os.path.basename(args.prompt_path).split('.')[0]

    # Load datasets
    datasets_per_lang = {}
    for lang in args.languages:
        datasets_per_lang[lang] = load_dataset("Davlan/sib200", lang, split="test")

    # Load hooked model
    print(f'Load model: {args.model_name}')
    saver = ActivationSaver(args.output_dir, task_id='topic_classification', model_name=args.model_name, prompt_id=prompt_id)
    hooked_model = HookedModel(args.model_name, saver=saver)  # Initialize with a hook fn saver
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for lang in args.languages:
        for instance in tqdm(datasets_per_lang[lang], desc=f"Processing activation for topic classification task ({lang})"):

            # Setup activation save location
            hooked_model.set_saver_id(str(instance['index_id']))
            hooked_model.set_saver_lang(lang)

            # Setup prompt
            prompt = initial_prompt.replace("{text}", instance['text'])

            # Add template for instruct models
            if args.is_base_model or 'bloom' in args.model_name:
                text = prompt
            else:
                messages = [{'role': 'user', 'content': prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # Turn off thinking for Qwen3 models
                )
            
            # Tokenize inputs
            inputs = tokenizer([text], return_tensors="pt").to(hooked_model.model.device)

            # Run forward prop
            _ = hooked_model.generate(inputs)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Extract activation for topic classification task")
	parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name")
	parser.add_argument("--prompt_path", type=str, default="./prompts/tc/prompt_en.txt", help="Path to the prompt file")
	parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
	parser.add_argument('--languages', type=str, nargs='+', default=['fra_Latn', 'eng_Latn', 'ind_Latn'], help='List of languages')
	parser.add_argument('--is_base_model', action='store_true', help='Whether the model is a base model or a instruct model')

	args = parser.parse_args()
	main(args)
