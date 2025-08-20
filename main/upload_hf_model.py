import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import PeftModel
import os
import argparse

def main(model_name, source_dir, target_dir, hf_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto")
            
    model = PeftModel.from_pretrained(
        model, 
        source_dir, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload", 
    )

    print("Merging and unloading model...")
    model = model.merge_and_unload()
    print("Saving model...")
    model.save_pretrained(target_dir)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(target_dir)
    print("Pushing model to hub...")
    model.push_to_hub(hf_name)
    tokenizer.push_to_hub(hf_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--hf_name", type=str, required=True)
    args = parser.parse_args()
    main(args.model_name, args.source_dir, args.target_dir, args.hf_name)