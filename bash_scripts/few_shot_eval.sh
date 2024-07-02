#!/bin/bash

# Define the dataset name
dataset_name="children"
LM_PATH="./LMs"

# Run the main Python script with specified parameters
python main.py \
  --seed 0 \
  --cut_off 128 \
  --eval_batch_size 256 \
  --num_roots 10 \
  --length 10 \
  --gnn_type gat \
  --eval_steps 10000 \
  --hidden_size 768 \
  --lm_type microsoft/deberta-base \
  --lm_path $LM_PATH \
  --dataset_name $dataset_name \
  --task nc \
  --eval_only True \
  --eval_model_path ./output/$dataset_name/model_deberta-base_2.pt \
  --few_shot \
  --few_shot_setting 3,3 \
  --k_qry 10 \
  --num_tasks 50 \
  --num_repeat 5 \
  --eval_num_tasks 50 \
  --prompt_epochs 50 \
  --token_num 3 \
  --label_as_init True \
  --LM_as_init True \
  --prompt_type only_first
