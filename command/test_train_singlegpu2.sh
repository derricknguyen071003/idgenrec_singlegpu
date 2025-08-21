#!/bin/bash

# Ensure CUDA is available and print status
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
# rounds: number of alternating training rounds (gen + rec count as one round)
# id_batch_size: batch size for GEN model
# rec_batch_size: batch size for Rec model
# sample_prompt: 0 if use all templates, 1 if use one particular template
# sample_num: number of samples to generate for each item
# max_his: maximum number of history items to consider (will be divided by 2 for id and rec)
# train: 1 for training, 0 for not training
python ../src/main_generative.py \
  --datasets Beauty \
  --distributed 0 \
  --gpu 0 \
  --tasks sequential \
  --item_indexing generative \
  --rounds 3 \
  --id_batch_size 1 \
  --rec_batch_size 1 \
  --prompt_file ../prompt.txt \
  --sample_prompt 1 \
  --eval_batch_size 1 \
  --sample_num 1 \
  --max_his 20 \
  --train 0 \
  --test_prompt seen:0 \
  --rec_lr 1e-3 \
  --id_lr 1e-8 \
  --test_epoch_id 1 \
  --test_epoch_rec 1 \
  --his_prefix 0 \
  --random_initialize 0 \
  --id_epochs 1 \
  --rec_epochs 10 \
  --alt_style id_first \
  --id_model_path /home/derrick/Documents/IDGenRec/model/Beauty_id_1_rec_10_20250617_122948/model_gen_round1_final.pt \
  --rec_model_path /home/derrick/Documents/IDGenRec/model/Beauty_id_1_rec_10_20250617_122948/model_rec_round1_final.pt