from transformers import AutoTokenizer

def create_tokenizer(model_name_or_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Can not find the tokenizer in local directory or huggingface hub
    except OSError:
        tokenizer = None
    
    return tokenizer
