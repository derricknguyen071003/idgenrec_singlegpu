item_gpu=${item_gpu:-3}
datasets=${datasets}
run_id=${run_id}
run_type=${run_type}
use_diffusion=${use_diffusion:-0}
social_quantization_id=${social_quantization_id:-0}
rounds=${rounds:-2}
train=${train:-1}
alt_style=${alt_style:-"id_first"}
use_wandb=${use_wandb:-1}
use_friend_seq=${use_friend_seq:-0}
random_remove_friend=${random_remove_friend:-0.0}
sample_num=${sample_num:-1}
socialtoid_mode=${socialtoid_mode:-"sequential"}  
social_sample_prompt=${social_sample_prompt:-1}
gradient_accumulation_steps=${gradient_accumulation_steps:-4}
lambda_mask=${lambda_mask:-1}
lambda_kl=${lambda_kl:-0.1}
echo "lambda_mask $lambda_mask and lambda_kl $lambda_kl"
# Optuna hyperparameter tuning
use_optuna=${use_optuna:-0}
optuna_n_trials=${optuna_n_trials:-20}
optuna_study_name=${optuna_study_name:-"idgenrec_optuna"}
optuna_storage=${optuna_storage:-""}
optuna_direction=${optuna_direction:-"maximize"}
optuna_pruner=${optuna_pruner:-"median"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if [[ "$datasets" == "yelp_30_2_5" ]]; then
  id_batch_size=8
  rec_batch_size=12
  social_batch_size=8
  echo "Using id_batch_size $id_batch_size and rec_batch_size $rec_batch_size and social_batch_size $social_batch_size"
fi
if [[ "$datasets" == "lastfm_full_10_10" || "$datasets" == "deli_full_10_10" || "$datasets" == "lastfm4i" || "$datasets" == "lastfm" ]]; then
    id_batch_size=16
    rec_batch_size=32
    social_batch_size=16
  echo "Using id_batch_size $id_batch_size and rec_batch_size $rec_batch_size and social_batch_size $social_batch_size"
fi

if [[ "$use_optuna" == "1" ]]; then
  # Run Optuna hyperparameter tuning
  python ../src/optuna_tune.py --datasets $datasets \
    --item_gpu $item_gpu \
    --run_id $run_id \
    --train $train \
    --alt_style $alt_style \
    --rounds $rounds \
    --run_type $run_type \
    --tasks sequential \
    --socialtoid_mode $socialtoid_mode \
    --item_indexing generative \
    --id_batch_size $id_batch_size \
    --rec_batch_size $rec_batch_size \
    --social_batch_size $social_batch_size \
    --prompt_file ../template/prompt.txt \
    --social_prompt_file ../template/prompt_social.txt \
    --sample_prompt 1 \
    --social_sample_prompt $social_sample_prompt \
    --eval_batch_size 1 \
    --dist_sampler 0 \
    --max_his 20  \
    --sample_num $sample_num \
    --use_friend_seq $use_friend_seq \
    --random_remove_friend $random_remove_friend \
    --id_epochs 1 \
    --rec_epochs 10 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --use_wandb $use_wandb \
    --social_quantization_id $social_quantization_id \
    --use_diffusion $use_diffusion \
    --lambda_mask $lambda_mask \
    --lambda_kl $lambda_kl \
    --n_trials $optuna_n_trials \
    --study_name $optuna_study_name \
    --direction $optuna_direction \
    --pruner $optuna_pruner \
    ${optuna_storage:+--storage $optuna_storage}
else
  python ../src/main_generative.py --datasets $datasets \
    --item_gpu $item_gpu \
    --run_id $run_id \
    --train $train \
    --alt_style $alt_style \
    --rounds $rounds \
    --run_type $run_type \
    --tasks sequential \
    --socialtoid_mode $socialtoid_mode \
    --item_indexing generative \
    --id_batch_size $id_batch_size \
    --rec_batch_size $rec_batch_size \
    --social_batch_size $social_batch_size \
    --prompt_file ../template/prompt.txt \
    --social_prompt_file ../template/prompt_social.txt \
    --sample_prompt 1 \
    --social_sample_prompt $social_sample_prompt \
    --eval_batch_size 1 \
    --dist_sampler 0 \
    --max_his 20  \
    --sample_num $sample_num \
    --use_friend_seq $use_friend_seq \
    --random_remove_friend $random_remove_friend \
    --rec_lr 1e-3 \
    --id_lr 1e-8 \
    --id_epochs 1 \
    --rec_epochs 10 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --use_wandb $use_wandb \
    --social_quantization_id $social_quantization_id \
    --use_diffusion $use_diffusion \
    --lambda_mask $lambda_mask \
    --lambda_kl $lambda_kl
fi