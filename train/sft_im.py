import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from transformers.utils import is_torch_xpu_available
os.environ["HF_USE_AUTO_TP"] = "1"

# ==== Patch: shared cache + output dirs ====
CACHE_DIR  = "/shared/share_mala/Ishaan/cache/qwen-s1"
OUTPUT_DIR = "/shared/share_mala/Ishaan/finetuned_model/qwen-s1"

# set HF cache envs
os.environ["HF_HOME"]            = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"]  = CACHE_DIR
os.environ["HF_MODULES_CACHE"]   = CACHE_DIR
os.environ["HF_METRICS_CACHE"]   = CACHE_DIR
# =========================================

from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="S1")
    wandb_entity: Optional[str] = field(default="ishaanmaheshwari2001-columbia-university")  # your WandB username

    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY']  = self.wandb_entity


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model (force cache_dir)
    kwargs = {}
    if "70B" in config.model_name:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            cache_dir=CACHE_DIR,
            **kwargs
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            cache_dir=CACHE_DIR
        )

    # load dataset (will use HF_DATASETS_CACHE)
    dataset = load_dataset(config.train_file_path)

    # tokenizer setup (force cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name,
        cache_dir=CACHE_DIR,
        use_fast=True
    )
    
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template    = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template    = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    # collator only computes loss over assistant responses
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    args.dataset_text_field = 'text'
    args.max_seq_length     = config.block_size
    args.deepspeed = "train/ds_config.json"
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    # train + save to shared OUTPUT_DIR
    trainer.train()
    trainer.save_model(output_dir=OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
