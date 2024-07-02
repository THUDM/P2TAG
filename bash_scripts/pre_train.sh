#!/bin/bash

# Define the dataset name
DT_NAME="arxiv"
LM_PATH="./LMs"

# Run the main Python script with specified parameters
python main.py \
  --seed 0 \
  --lr 1e-5 \
  --mask_rate 0.75  \
  --batch_size 100 \
  --cut_off 128 \
  --eval_batch_size 256 \
  --num_roots 10 \
  --length 10 \
  --num_epochs 3 \
  --gnn_type gat \
  --eval_steps 10000 \
  --hidden_size 768 \
  --lm_type microsoft/deberta-base \
  --lm_path $LM_PATH \
  --dataset_name $DT_NAME \
  --process_mode TA \
  --task nc \
  --save_model_path ./output/$DT_NAME/ \
  --eval_only False \
