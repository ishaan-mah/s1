import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# -------------------
# CONFIG
# -------------------
model_id = "Qwen/Qwen2.5-32B-Instruct"
cache_dir = "/shared/share_mala/Ishaan/cache/qwen-s1/"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 8
max_input_tokens = 2048
dataset_name = "simplescaling/aime24_nofigures"

# -------------------
# Prompt Format
# -------------------
def build_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. Answer with only the final number. Do not explain.<|im_end|>\n"
        f"<|im_start|>user\n{problem.strip()}\nFinal Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )



def extract_numeric_answer(text: str) -> str:
    match = re.search(r"\b\d+\b", text)
    return match.group(0) if match else "NA"


# -------------------
# Evaluation Function
# -------------------
def evaluate_model():
    import logging
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    dataset = load_dataset(dataset_name, split="train")  # fixed from 'test'
    total, correct = 0, 0

    print(f"🧪 Evaluating {len(dataset)} problems...")

    for i, example in enumerate(tqdm(dataset)):
        prompt = build_prompt(example["problem"])
        gold = str(example["answer"]).strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        pred = extract_numeric_answer(decoded)

        if pred == gold:
            correct += 1
        else:
            print(f"\n❌ Mismatch [Example {i}]")
            print(f"🔹Prompt:\n{prompt}")
            print(f"🔹Decoded Output:\n{decoded}")
            print(f"🔹Predicted Answer: {pred}")
            print(f"🔹Correct Answer:   {gold}")

        total += 1

    acc = correct / total
    print(f"\n✅ Base Qwen Accuracy: {acc*100:.2f}% ({correct}/{total})")

# -------------------
# Run
# -------------------
if __name__ == "__main__":
    evaluate_model()
