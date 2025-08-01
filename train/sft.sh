# Reference Running: bash train/sft.sh
export MASTER_PORT=39501
export TRITON_CACHE_DIR="/tmp/$USER/triton-cache"
export HF_USE_AUTO_TP=1
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-32B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=2 # requires more GPU memory
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)
push_to_hub=false
min_lr=0

deepspeed --master_port ${MASTER_PORT} --num_gpus=8 train/sft.py \
    --deepspeed train/ds_config.json \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=5 \
    --train_file_path="/shared/share_mala/Ishaan/s1k_mixed_Data" \
    --model_name="Qwen/Qwen2.5-32B-Instruct" \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_strategy="epoch" \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="/shared/share_mala/Ishaan/finetuned_model/qwen-s1" \
    --push_to_hub False \
    --save_only_model=True