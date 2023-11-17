from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_DIR = Path("models")
models = ["gpt2", "facebook/opt-125m", "EleutherAI/pythia-70m"]

def download_model(model_name):
    """Download a Huggingface model and tokenizer to the specified directory"""
    # create directory if necessary
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir()
    model_path = MODEL_DIR.joinpath(model_name)

    # download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

for model in models:
    download_model(model)