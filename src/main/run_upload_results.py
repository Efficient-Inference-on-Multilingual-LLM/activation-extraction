import os
import glob
from huggingface_hub import HfApi
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv() 

# Login HuggingFace
login(token=os.getenv("HF_TOKEN"))
api = HfApi()

# Define your repository details, the outputs parent directory, task, models, and prompt configuration (* if all)
repo_id = "indolinguafrancaresearch/extracted-activations"  
parent_directory = "outputs"
task = 'topic_classification'
models = '*'
prompt_config = 'prompted'

# Get all compressed experiment results (activations)
files_to_upload = glob.glob(os.path.join(parent_directory, task, models, prompt_config, '*.tar.gz'))

# Loop through and upload each file
for file_path in files_to_upload:
    if os.path.isfile(file_path):
        print(f"Uploading {file_path}...")
        api.upload_file(
            path_or_fileobj=file_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=file_path,  # Creates a folder with the same name in the repo
        )
        print(f"Successfully uploaded {file_path}.")

print("All specified files have been uploaded.")