#!/bin/bash

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7  # Adjust based on your available GPUs

model_paths=(
    "./checkpoints/ganga-2-1b-embeddings-new-equall-finetune-final-2-mean-32-epoch-1/best_model"
    "./checkpoints/ganga-2-1b-embeddings-new-equall-finetune-final-mean-32-epoch-1/best_model"
)

# Loop through each model and run the test
for model_path in "${model_paths[@]}"
do
    echo "Running MTEB test on model: $model_path"
    python mteb_test.py --model_name "$model_path"
done

python mteb_leaderboard.py