#!/bin/bash

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your available GPUs
export TOKENIZERS_PARALLELISM=true

# Create directories
mkdir -p logs checkpoints

# Set training parameters
MODEL_NAME="LingoIITGN/Ganga-2-1B"
BATCH_SIZE=42  # Small batch size for full model fine-tuning
MAX_LENGTH=512
LEARNING_RATE=2e-6  # Lower learning rate for full model fine-tuning
NUM_EPOCHS=1
DATA_DIR="./new_training_data"
OUTPUT_DIR="./checkpoints/ganga-2-1b-embeddings"
ZERO_STAGE=3

# Enable mixed precision but no LoRA
USE_FP16="--use_fp16"  # Mixed precision training

# Number of GPUs to use
NUM_GPUS=1

# Log start time and configuration
echo "Starting training at $(date)"
echo "Using model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"


deepspeed --include=localhost:0 \
 trainer_with_deepspeed.py \
  --model_name $MODEL_NAME \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --learning_rate $LEARNING_RATE \
  --warmup_steps 100 \
  --num_epochs $NUM_EPOCHS \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --pooling_type "selective" \
  --temperature 0.2 \
  --weight_decay 0.001 \
  --logging_steps 1 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --deepspeed \
  --deepspeed_stage $ZERO_STAGE \
  --local_rank 0 \
  --num_gpus $NUM_GPUS \
  --model_parallel_size 1 \
  --hinvec_selective \
  --use_wandb \
  2>&1 | tee logs/training_log_$(date +%Y%m%d_%H%M%S).log

# Log end time
echo "Training completed at $(date)"

# Check if training completed successfully
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed with error code $?"
fi