import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import numpy as np
from peft import PeftModel

def main(num):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    peft_model = f"/data/user_data/gyeongwk/KernelBench/sft/sft_Qwen2.5-7B-Instruct_best_{num}"
    merged_peft_model_name = f"gyeongwk/sft_Qwen2.5-7B-Instruct_best_{num}-merged-peft"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    max_seq_length = tokenizer.model_max_length

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto")
            
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload", 
    )

    model = model.merge_and_unload()
    model.save_pretrained(merged_peft_model_name)
    tokenizer.save_pretrained(merged_peft_model_name)
    model.push_to_hub(merged_peft_model_name)
    tokenizer.push_to_hub(merged_peft_model_name)


if __name__ == "__main__":
    for i in range(1, 5):
        main(i)