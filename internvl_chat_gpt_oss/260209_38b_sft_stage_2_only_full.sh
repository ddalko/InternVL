#!/bin/bash
#SBATCH --job-name=260209_sft_stage_2_only_full
#SBATCH --partition=compute
#SBATCH --nodelist=compute-st-kait-gpu-[1-8]
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1536G

set -euo pipefail

export TF_CPP_MIN_LOG_LEVEL=3
export USE_TCS_LOADER=0
export LAUNCHER=pytorch

# ============================================================================
# Distributed Training Configuration
# ============================================================================
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR

# Debugging options (uncomment if needed)
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Training parameters
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}  # Target global batch size (8 node × 8 GPUs × 1 per-device × 8 grad_acc = 512)
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}

# Calculate gradient accumulation steps to achieve target global batch size
# Global batch = per_device_batch × grad_acc × num_gpus
# grad_acc = global_batch / (per_device_batch × num_gpus)
TOTAL_GPUS=$((SLURM_NNODES * NPROC_PER_NODE))
GRADIENT_ACC=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / TOTAL_GPUS))
GRADIENT_ACC=$((GRADIENT_ACC > 0 ? GRADIENT_ACC : 1))  # Ensure at least 1

echo "=== Batch Size Configuration ==="
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Per-device Batch Size: $PER_DEVICE_BATCH_SIZE"  
echo "Gradient Accumulation Steps: $GRADIENT_ACC"
echo "Total GPUs: $TOTAL_GPUS (${SLURM_NNODES} nodes × ${NPROC_PER_NODE} GPUs)"
echo "Effective Global Batch: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACC * TOTAL_GPUS))"
echo "================================"

# ============================================================================
# Project and Directory Configuration
# ============================================================================
# Project settings
PROJECT_NAME=${SLURM_JOB_NAME}

# Output directories
# EXP_NAME=${EXP_NAME:-$SLURM_JOB_NAME}
EXP_NAME=260209_38b_sft_stage_2_only_full
export OUTPUT_DIR=work_dirs/${PROJECT_NAME}/${EXP_NAME}
OUTPUT_DIR=work_dirs/${PROJECT_NAME}/${EXP_NAME}
export TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
export JOBLOG=${OUTPUT_DIR}/training.log

# Log directory
timestamp=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/${timestamp}"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGDIR"

# ============================================================================
# Environment Setup
# ============================================================================
cd /fsx/VLM/internvl3_5/code/InternVL-debug/internvl_chat_gpt_oss

echo "HOSTNAME=$(hostname)"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "SLURM_NODEID=$SLURM_NODEID SLURM_PROCID=$SLURM_PROCID SLURM_LOCALID=$SLURM_LOCALID"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# ============================================================================
# Launch Distributed Training
# ============================================================================

CONV_STYLE=internvl2_5
export CONV_STYLE

srun --ntasks-per-node=1 \
  --output="$LOGDIR/%x_%j_%N.out" \
  --error="$LOGDIR/%x_%j_%N.err" \
  bash -c '
    echo "[srun] HOSTNAME=$(hostname) SLURM_PROCID=$SLURM_PROCID SLURM_LOCALID=$SLURM_LOCALID"
    source /fsx/VLM/envs/internvl3.5/bin/activate
    export PYTHONPATH="/fsx/VLM/internvl3_5/code/InternVL-debug/internvl_chat_gpt_oss:${PYTHONPATH:-}"
    export TORCH_EXTENSIONS_DIR=/fsx/VLM/envs/internvl3.5/torch_extensions
    export TRITON_CACHE_DIR=/fsx/VLM/envs/internvl3.5/triton_cache

    torchrun \
      --nnodes="$SLURM_NNODES" \
      --node-rank="$SLURM_PROCID" \
      --master-addr="'"$MASTER_ADDR"'" \
      --master-port="'"$MASTER_PORT"'" \
      --nproc-per-node="'"$NPROC_PER_NODE"'" \
      internvl/train/internvl_chat_finetune.py \
      --model_name_or_path "OpenGVLab/InternVL3_5-38B-Pretrained" \
      --conv_style "'"$CONV_STYLE"'" \
      --use_fast_tokenizer False \
      --output_dir "'"$OUTPUT_DIR"'" \
      --meta_path "./shell/data/sft_stage2_4_6.json" \
      --overwrite_output_dir True \
      --force_image_size 448 \
      --max_dynamic_patch 12 \
      --down_sample_ratio 0.5 \
      --drop_path_rate 0.0 \
      --min_num_frame 8 \
      --max_num_frame 32 \
      --freeze_llm False \
      --freeze_mlp False \
      --freeze_backbone False \
      --vision_select_layer -1 \
      --dataloader_num_workers 1 \
      --bf16 True \
      --max_steps 42000 \
      --per_device_train_batch_size '"$PER_DEVICE_BATCH_SIZE"' \
      --gradient_accumulation_steps '"$GRADIENT_ACC"' \
      --save_strategy "steps" \
      --save_steps 1000 \
      --save_total_limit 3 \
      --learning_rate 5e-6 \
      --weight_decay 0.05 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 50 \
      --max_seq_length 16384 \
      --split_annotations True \
      --do_train True \
      --grad_checkpoint True \
      --gradient_checkpointing True \
      --group_by_length False \
      --dynamic_image_size True \
      --use_thumbnail True \
      --ps_version "v2" \
      --use_custom_flash_attn False \
      --report_to "tensorboard" \
      --deepspeed "zero_stage1_config.json" \
      --log_freq 1000 \
      --seed 42 \
      2>&1 | tee -a "'"$OUTPUT_DIR"'/training_log.txt"
  '