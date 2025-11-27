#!/bin/bash

# Extensive Grid Search Hyperparameter Fine-tuning Script
# This script performs exhaustive grid search over hyperparameter combinations

# ============================================================================
# Configuration - Modify these values to customize the grid search
# ============================================================================

# Base configuration (required)
item_gpu=${item_gpu:-3}
datasets=${datasets:-"lastfm4i"}
run_type=${run_type:-"original_idgenrec"}
use_diffusion=${use_diffusion:-1}
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

# Base run ID prefix (will append grid search parameters)
base_run_id=${base_run_id:-"grid_search"}

# Grid search configuration
# Define arrays of values to search over

# Learning rates (log scale)
id_lr_values=(1e-8 1e-7 1e-6 1e-5)
rec_lr_values=(1e-5 1e-4 1e-3)

# Diffusion hyperparameters (only used if use_diffusion=1)
if [[ "$use_diffusion" == "1" ]]; then
    lambda_mask_values=(0.01 0.1 1.0)
    lambda_kl_values=(0.01 0.1 1.0)
    diffusion_beta_max_values=(0.05 0.1 0.2)
    diffusion_cross_prob_values=(0.3 0.5 0.7)
    noise_head_dropout_values=(0.0 0.1 0.2)
else
    # Empty arrays if diffusion is disabled
    lambda_mask_values=()
    lambda_kl_values=()
    diffusion_beta_max_values=()
    diffusion_cross_prob_values=()
    noise_head_dropout_values=()
fi

# Training hyperparameters
weight_decay_values=(0.001 0.01 0.1)
warmup_prop_values=(0.05 0.1 0.15)
gradient_accumulation_steps_values=(2 4 8)
clip_values=(0.5 1.0 2.0)

# ============================================================================
# Setup
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Set batch sizes based on dataset
if [[ "$datasets" == "yelp_30_2_5" ]]; then
  id_batch_size=8
  rec_batch_size=12
  social_batch_size=8
  echo "Using id_batch_size $id_batch_size and rec_batch_size $rec_batch_size and social_batch_size $social_batch_size"
elif [[ "$datasets" == "lastfm_full_10_10" || "$datasets" == "deli_full_10_10" || "$datasets" == "lastfm4i" ]]; then
    id_batch_size=16
    rec_batch_size=32
    social_batch_size=16
  echo "Using id_batch_size $id_batch_size and rec_batch_size $rec_batch_size and social_batch_size $social_batch_size"
else
  # Default batch sizes
  id_batch_size=16
  rec_batch_size=32
  social_batch_size=16
fi

# Create results directory
results_dir="grid_search_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$results_dir"
summary_file="$results_dir/grid_search_summary.txt"

# Initialize summary file
echo "Grid Search Hyperparameter Tuning Summary" > "$summary_file"
echo "==========================================" >> "$summary_file"
echo "Dataset: $datasets" >> "$summary_file"
echo "Run Type: $run_type" >> "$summary_file"
echo "Use Diffusion: $use_diffusion" >> "$summary_file"
echo "Start Time: $(date)" >> "$summary_file"
echo "" >> "$summary_file"
echo "Grid Configuration:" >> "$summary_file"
echo "  id_lr: ${id_lr_values[@]}" >> "$summary_file"
echo "  rec_lr: ${rec_lr_values[@]}" >> "$summary_file"
if [[ "$use_diffusion" == "1" ]]; then
    echo "  lambda_mask: ${lambda_mask_values[@]}" >> "$summary_file"
    echo "  lambda_kl: ${lambda_kl_values[@]}" >> "$summary_file"
    echo "  diffusion_beta_max: ${diffusion_beta_max_values[@]}" >> "$summary_file"
    echo "  diffusion_cross_prob: ${diffusion_cross_prob_values[@]}" >> "$summary_file"
    echo "  noise_head_dropout: ${noise_head_dropout_values[@]}" >> "$summary_file"
fi
echo "  weight_decay: ${weight_decay_values[@]}" >> "$summary_file"
echo "  warmup_prop: ${warmup_prop_values[@]}" >> "$summary_file"
echo "  gradient_accumulation_steps: ${gradient_accumulation_steps_values[@]}" >> "$summary_file"
echo "  clip: ${clip_values[@]}" >> "$summary_file"
echo "" >> "$summary_file"
echo "Results:" >> "$summary_file"
echo "--------" >> "$summary_file"
printf "%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
    "ID" "id_lr" "rec_lr" "lambda_mask" "lambda_kl" "beta_max" "cross_prob" "noise_drop" "weight_decay" "warmup" "grad_acc" "clip" "Status" "Time" >> "$summary_file"
echo "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" >> "$summary_file"

# ============================================================================
# Grid Search Loop
# ============================================================================

config_id=0
total_configs=0
successful_configs=0
failed_configs=0

# Calculate total number of configurations
if [[ "$use_diffusion" == "1" ]]; then
    total_configs=$((${#id_lr_values[@]} * ${#rec_lr_values[@]} * ${#lambda_mask_values[@]} * ${#lambda_kl_values[@]} * \
                     ${#diffusion_beta_max_values[@]} * ${#diffusion_cross_prob_values[@]} * ${#noise_head_dropout_values[@]} * \
                     ${#weight_decay_values[@]} * ${#warmup_prop_values[@]} * ${#gradient_accumulation_steps_values[@]} * ${#clip_values[@]}))
else
    total_configs=$((${#id_lr_values[@]} * ${#rec_lr_values[@]} * ${#weight_decay_values[@]} * \
                     ${#warmup_prop_values[@]} * ${#gradient_accumulation_steps_values[@]} * ${#clip_values[@]}))
fi

echo "Starting grid search with $total_configs total configurations..."
echo "This may take a very long time. Results will be saved to $results_dir/"

# Nested loops for grid search
for id_lr in "${id_lr_values[@]}"; do
for rec_lr in "${rec_lr_values[@]}"; do

if [[ "$use_diffusion" == "1" ]]; then
    for lambda_mask in "${lambda_mask_values[@]}"; do
    for lambda_kl in "${lambda_kl_values[@]}"; do
    for diffusion_beta_max in "${diffusion_beta_max_values[@]}"; do
    for diffusion_cross_prob in "${diffusion_cross_prob_values[@]}"; do
    for noise_head_dropout in "${noise_head_dropout_values[@]}"; do
    for weight_decay in "${weight_decay_values[@]}"; do
    for warmup_prop in "${warmup_prop_values[@]}"; do
    for gradient_accumulation_steps in "${gradient_accumulation_steps_values[@]}"; do
    for clip in "${clip_values[@]}"; do
        config_id=$((config_id + 1))
        
        # Create unique run_id for this configuration
        run_id="${base_run_id}_${config_id}_idlr${id_lr}_reclr${rec_lr}_lm${lambda_mask}_lk${lambda_kl}_bm${diffusion_beta_max}_cp${diffusion_cross_prob}_nd${noise_head_dropout}_wd${weight_decay}_wp${warmup_prop}_ga${gradient_accumulation_steps}_c${clip}"
        
        echo ""
        echo "========================================================================"
        echo "Configuration $config_id/$total_configs: $run_id"
        echo "========================================================================"
        echo "Parameters:"
        echo "  id_lr: $id_lr"
        echo "  rec_lr: $rec_lr"
        echo "  lambda_mask: $lambda_mask"
        echo "  lambda_kl: $lambda_kl"
        echo "  diffusion_beta_max: $diffusion_beta_max"
        echo "  diffusion_cross_prob: $diffusion_cross_prob"
        echo "  noise_head_dropout: $noise_head_dropout"
        echo "  weight_decay: $weight_decay"
        echo "  warmup_prop: $warmup_prop"
        echo "  gradient_accumulation_steps: $gradient_accumulation_steps"
        echo "  clip: $clip"
        echo ""
        
        start_time=$(date +%s)
        
        # Run training
        if python ../src/main_generative.py \
            --datasets $datasets \
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
            --max_his 20 \
            --sample_num $sample_num \
            --use_friend_seq $use_friend_seq \
            --random_remove_friend $random_remove_friend \
            --id_lr $id_lr \
            --rec_lr $rec_lr \
            --id_epochs 1 \
            --rec_epochs 10 \
            --weight_decay $weight_decay \
            --warmup_prop $warmup_prop \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --clip $clip \
            --use_wandb $use_wandb \
            --social_quantization_id $social_quantization_id \
            --use_diffusion $use_diffusion \
            --lambda_mask $lambda_mask \
            --lambda_kl $lambda_kl \
            --diffusion_beta_max $diffusion_beta_max \
            --diffusion_cross_prob $diffusion_cross_prob \
            --noise_head_dropout $noise_head_dropout \
            > "$results_dir/${run_id}.log" 2>&1; then
            
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            status="SUCCESS"
            successful_configs=$((successful_configs + 1))
            echo "✓ Configuration $config_id completed successfully in ${duration}s"
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            status="FAILED"
            failed_configs=$((failed_configs + 1))
            echo "✗ Configuration $config_id failed after ${duration}s"
        fi
        
        # Append to summary
        printf "%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
            "$config_id" "$id_lr" "$rec_lr" "$lambda_mask" "$lambda_kl" "$diffusion_beta_max" \
            "$diffusion_cross_prob" "$noise_head_dropout" "$weight_decay" "$warmup_prop" \
            "$gradient_accumulation_steps" "$clip" "$status" "${duration}s" >> "$summary_file"
        
    done  # clip
    done  # gradient_accumulation_steps
    done  # warmup_prop
    done  # weight_decay
    done  # noise_head_dropout
    done  # diffusion_cross_prob
    done  # diffusion_beta_max
    done  # lambda_kl
    done  # lambda_mask
else
    # Non-diffusion mode
    for weight_decay in "${weight_decay_values[@]}"; do
    for warmup_prop in "${warmup_prop_values[@]}"; do
    for gradient_accumulation_steps in "${gradient_accumulation_steps_values[@]}"; do
    for clip in "${clip_values[@]}"; do
        config_id=$((config_id + 1))
        
        # Create unique run_id for this configuration
        run_id="${base_run_id}_${config_id}_idlr${id_lr}_reclr${rec_lr}_wd${weight_decay}_wp${warmup_prop}_ga${gradient_accumulation_steps}_c${clip}"
        
        echo ""
        echo "========================================================================"
        echo "Configuration $config_id/$total_configs: $run_id"
        echo "========================================================================"
        echo "Parameters:"
        echo "  id_lr: $id_lr"
        echo "  rec_lr: $rec_lr"
        echo "  weight_decay: $weight_decay"
        echo "  warmup_prop: $warmup_prop"
        echo "  gradient_accumulation_steps: $gradient_accumulation_steps"
        echo "  clip: $clip"
        echo ""
        
        start_time=$(date +%s)
        
        # Run training
        if python ../src/main_generative.py \
            --datasets $datasets \
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
            --max_his 20 \
            --sample_num $sample_num \
            --use_friend_seq $use_friend_seq \
            --random_remove_friend $random_remove_friend \
            --id_lr $id_lr \
            --rec_lr $rec_lr \
            --id_epochs 1 \
            --rec_epochs 10 \
            --weight_decay $weight_decay \
            --warmup_prop $warmup_prop \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --clip $clip \
            --use_wandb $use_wandb \
            --social_quantization_id $social_quantization_id \
            --use_diffusion $use_diffusion \
            > "$results_dir/${run_id}.log" 2>&1; then
            
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            status="SUCCESS"
            successful_configs=$((successful_configs + 1))
            echo "✓ Configuration $config_id completed successfully in ${duration}s"
        else
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            status="FAILED"
            failed_configs=$((failed_configs + 1))
            echo "✗ Configuration $config_id failed after ${duration}s"
        fi
        
        # Append to summary
        printf "%-6s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n" \
            "$config_id" "$id_lr" "$rec_lr" "$weight_decay" "$warmup_prop" \
            "$gradient_accumulation_steps" "$clip" "$status" "${duration}s" >> "$summary_file"
        
    done  # clip
    done  # gradient_accumulation_steps
    done  # warmup_prop
    done  # weight_decay
fi

done  # rec_lr
done  # id_lr

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "Grid Search Complete!"
echo "========================================================================"
echo "Total configurations: $total_configs"
echo "Successful: $successful_configs"
echo "Failed: $failed_configs"
echo ""
echo "Results saved to: $results_dir/"
echo "Summary file: $summary_file"
echo "End Time: $(date)" >> "$summary_file"
echo "" >> "$summary_file"
echo "Final Statistics:" >> "$summary_file"
echo "  Total configurations: $total_configs" >> "$summary_file"
echo "  Successful: $successful_configs" >> "$summary_file"
echo "  Failed: $failed_configs" >> "$summary_file"

