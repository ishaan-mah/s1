import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -------------------
# CONFIG
# -------------------

models = {
    "s1_finetuned": "/shared/share_mala/Ishaan/finetuned_model/qwen-s1(32B)(s1)/merged",
    "s1k_mixed": "/shared/share_mala/Ishaan/finetuned_model/qwen-s1(32B)(s1k_mixed)/merged"
}
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 32
max_input_tokens = 2048
dataset_name = "simplescaling/aime24_nofigures"

# -------------------
# Prompt Format
# -------------------
def build_prompt(problem: str) -> str:
    return f"""<|im_start|>system
You are a helpful assistant that answers with only one number and no explanation.<|im_end|>
<|im_start|>user
{problem.strip()}<|im_end|>
<|im_start|>assistant
"""

def extract_numeric_answer(text: str) -> str:
    """
    Extracts the first integer from the assistant's response.
    """
    match = re.search(r"\b\d+\b", text)
    return match.group(0) if match else "NA"

# -------------------
# Evaluation Function
# -------------------
def evaluate_model(path, label):
    print(f"\n--- Evaluating: {label} ---")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    dataset = load_dataset(dataset_name, split="train")
    total, correct = 0, 0

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
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

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = extract_numeric_answer(decoded)

        if pred == gold:
            correct += 1
        else:
            print(f"\n❌ Mismatch [Example {idx}]")
            print("🔹Prompt:\n", prompt)
            print("🔹Decoded Output:\n", decoded.strip())
            print("🔹Predicted Answer:", pred)
            print("🔹Correct Answer:  ", gold)

        total += 1

    acc = correct / total
    print(f"\n✅ {label} Accuracy: {acc*100:.2f}% ({correct}/{total})")

# -------------------
# Run
# -------------------
if __name__ == "__main__":
    for label, path in models.items():
        evaluate_model(path, label)
