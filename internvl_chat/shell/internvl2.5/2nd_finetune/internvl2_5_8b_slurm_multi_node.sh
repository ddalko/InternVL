#!/bin/bash
#SBATCH --job-name=internvl2.5_8b_multi
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=0
#SBATCH --output=/fsx/VLM/internvl3/logs/slurm/internvl2_5_8b_multinode_%j.log
#SBATCH --error=/fsx/VLM/internvl3/logs/slurm/internvl2_5_8b_multinode_%j.err

# ============================================
# InternVL2.5-8B Multi-Node Training
# 4 nodes × 8 GPUs = 32 GPUs
# ============================================

set -x

mkdir -p /fsx/VLM/internvl3/logs/slurm

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Total GPUs: $(($SLURM_NNODES * 8))"
echo "Start time: $(date)"
echo "=========================================="

# ============================================
# Configuration
# ============================================
export TOTAL_GPUS=$(($SLURM_NNODES * 8))
export PER_DEVICE_BATCH_SIZE=4
export BATCH_SIZE=512  # 4 nodes * 8 GPUs * 4 batch * 4 grad_acc = 512
export GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / TOTAL_GPUS))

# Environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# Multi-node
export WORLD_SIZE=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export NPROC_PER_NODE=8

# NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=1800

# DeepSpeed
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0

# Output
cd /fsx/VLM/internvl3_5/code/InternVL-debug/internvl_chat
OUTPUT_DIR='work_dirs/internvl_chat_v2_5/internvl2_5_8b_multinode_4nodes'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

echo "=========================================="
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Batch Size (global): $BATCH_SIZE"
echo "  Per Device Batch: $PER_DEVICE_BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACC"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "=========================================="

# ============================================
# Training
# ============================================
srun --jobid=$SLURM_JOB_ID \
     --nodes=$SLURM_JOB_NUM_NODES \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --gpus-per-node=$SLURM_GPUS_PER_NODE \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash -c "export RANK=\$SLURM_PROCID && \
              export LOCAL_RANK=\$SLURM_LOCALID && \
              export WORLD_SIZE=\$SLURM_NTASKS && \
              python -u internvl/train/internvl_chat_finetune.py \
  --model_name_or_path OpenGVLab/InternVL2_5-8B \
  --conv_style internvl2_5 \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ./shell/data/internvl_1_2_finetune_custom.json \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version v2 \
  --deepspeed zero_stage1_config.json \
  --report_to tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
echo "Job completed: $EXIT_CODE"
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
