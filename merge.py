import os
import torch
import argparse
from safetensors.torch import save_file as safe_save_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

def find_model_shards(checkpoint_dir):
    # Finds DeepSpeed-style checkpoint files
    return sorted([
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith("_model_states.pt")
    ])

def merge_state_dicts(shard_files):
    print(f"[INFO] Only loading first shard: {os.path.basename(shard_files[0])}")
    shard = torch.load(shard_files[0], map_location="cpu")
    return shard["module"] if "module" in shard else shard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Directory containing FSDP model shards")
    parser.add_argument("--out_dir", required=True, help="Directory to save merged model")
    parser.add_argument("--base_model", required=True, help="HF base model ID (e.g. Qwen/Qwen2.5-32B-Instruct)")
    parser.add_argument("--use_safetensors", action="store_true", help="Save using safetensors")
    args = parser.parse_args()

    # Step 1: Init empty model
    print("[INFO] Loading base model config and initializing empty model...")
    config = AutoConfig.from_pretrained(args.base_model)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Step 2: Merge checkpoints
    shard_paths = find_model_shards(args.in_dir)
    state_dict = merge_state_dicts(shard_paths)
    model.load_state_dict(state_dict, strict=True)

    # Step 3: Save model
    print(f"[INFO] Saving merged model to {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.use_safetensors:
        safe_save_file(model.state_dict(), os.path.join(args.out_dir, "model.safetensors"))
    else:
        torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))

    # Save config and tokenizer
    model.config.save_pretrained(args.out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.out_dir)

    print("✅ Merge complete!")

if __name__ == "__main__":
    main()
