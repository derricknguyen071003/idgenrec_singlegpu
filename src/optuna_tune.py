"""
Minimal Optuna script to tune lambda_mask and lambda_kl parameters.
Supports both single-objective and multi-objective optimization.

Usage examples:
  # Multi-objective: optimize NDCG and Hit@5 and @10 (default)
  python src/optuna_tune.py --datasets lastfm4i --run_id optuna_tune --n_trials 20
  
  # Single metric
  python src/optuna_tune.py --datasets lastfm4i --run_id optuna_tune --n_trials 20 --optimize_metrics ndcg@10
  
  # Custom metrics
  python src/optuna_tune.py --datasets lastfm4i --run_id optuna_tune --n_trials 20 --optimize_metrics ndcg@5,hit@10
"""
import optuna
import argparse
import logging
import sys
import os
import json
import re
from pathlib import Path
from utils import utils
from main_generative import single_main
import torch

def extract_metrics_from_log(log_file_path, metrics_list):
    """Extract evaluation metrics from log file.
    
    Metrics are logged as numbers in order: hit@1, hit@5, hit@10, hit@20, ndcg@1, ndcg@5, ndcg@10, ndcg@20
    They appear after "Testing Item Recommender Performance" or "Testing Friend Recommender Performance"
    """
    if not os.path.exists(log_file_path):
        return None
    
    metrics_dict = {}
    in_testing_section = False
    metrics_found = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Look for testing section and extract metrics
    for i, line in enumerate(lines):
        # Check if we're entering a testing section
        if "Testing" in line and "Performance" in line:
            in_testing_section = True
            metrics_found = []
            continue
        
        # In testing section, look for metric values (numbers like 0.123)
        if in_testing_section:
            # Metrics are logged as simple numbers: "0.123" or "INFO - 0.123"
            # They appear on consecutive lines after "Testing ... Performance"
            match = re.search(r'(\d+\.\d{3})', line)
            if match:
                metrics_found.append(float(match.group(1)))
                # If we've collected all metrics, process them
                if len(metrics_found) == len(metrics_list):
                    # Map metrics to their names
                    for idx, metric_name in enumerate(metrics_list):
                        metrics_dict[metric_name] = metrics_found[idx]
                    in_testing_section = False
                    metrics_found = []
            # Reset if we see a non-metric line after starting
            elif "---" in line or "Finished" in line:
                if len(metrics_found) > 0:
                    # Process partial metrics if any
                    for idx, metric_name in enumerate(metrics_list[:len(metrics_found)]):
                        metrics_dict[metric_name] = metrics_found[idx]
                in_testing_section = False
                metrics_found = []
    
    # Also try to find metrics in a simpler pattern (just consecutive numbers)
    if not metrics_dict:
        # Look for patterns like: "0.123\n0.456\n0.789..." which are metrics
        all_numbers = []
        for line in lines:
            match = re.search(r'(\d+\.\d{3})', line)
            if match:
                all_numbers.append((float(match.group(1)), line))
        
        # If we found enough numbers, try to match them to metrics
        # Look for sequences of numbers that match the expected metric count
        if len(all_numbers) >= len(metrics_list):
            # Take the last occurrence of metrics (from final evaluation)
            # Metrics are logged together, so find a sequence
            for i in range(len(all_numbers) - len(metrics_list) + 1):
                candidate = [num for num, _ in all_numbers[i:i+len(metrics_list)]]
                # Check if these look like metrics (all between 0 and 1, or reasonable ranges)
                if all(0 <= m <= 1 for m in candidate) or all(0 <= m <= 100 for m in candidate):
                    for idx, metric_name in enumerate(metrics_list):
                        metrics_dict[metric_name] = candidate[idx]
                    break
    
    return metrics_dict if metrics_dict else None

def get_metric_index(metric_name, metrics_list):
    """Get the index of a metric in the metrics list"""
    try:
        return metrics_list.index(metric_name.lower())
    except ValueError:
        # Try case-insensitive match
        for i, m in enumerate(metrics_list):
            if m.lower() == metric_name.lower():
                return i
        return None

def objective(trial, args):
    """Optuna objective function that trains and evaluates the model"""
    
    # Suggest lambda_mask and lambda_kl values
    # Using log-uniform distribution for better exploration
    lambda_mask = trial.suggest_float("lambda_mask", 0.01, 10.0, log=True)
    lambda_kl = trial.suggest_float("lambda_kl", 0.01, 2.0, log=True)
    
    logging.info(f"Trial {trial.number}: lambda_mask={lambda_mask:.4f}, lambda_kl={lambda_kl:.4f}")
    
    # Update args with suggested values
    args.lambda_mask = lambda_mask
    args.lambda_kl = lambda_kl
    
    # Create unique run_id for this trial
    original_run_id = args.run_id
    args.run_id = f"{original_run_id}_trial{trial.number}"
    
    # Determine log file paths
    log_dir = Path(args.log_dir) if hasattr(args, 'log_dir') else Path("../log")
    train_log_file = log_dir / "train" / args.datasets / f"{args.run_id}.log"
    test_log_file = log_dir / "test" / args.datasets / f"{args.run_id}.log"
    
    try:
        # Run training (this will call single_main which trains and validates)
        original_train = args.train
        args.train = 1  # Ensure training happens
        single_main()
        
        # Run evaluation to get metrics
        # First, find the model checkpoint that was just saved
        model_dir = Path(args.model_dir) if hasattr(args, 'model_dir') else Path("../model")
        model_path = model_dir / args.datasets / args.run_id
        
        # Find the latest model checkpoint
        rec_model_path = None
        if model_path.exists():
            # Look for model files (try different patterns)
            model_files = list(model_path.glob("model_rec_item_round*_final.pt"))
            if not model_files:
                # Try alternative pattern
                model_files = list(model_path.glob("model_rec*_final.pt"))
            if not model_files:
                # Try any .pt file
                model_files = list(model_path.glob("*.pt"))
            
            if model_files:
                # Get the latest round or most recent file
                try:
                    rec_model_path = str(sorted(model_files, 
                        key=lambda x: int(re.search(r'round(\d+)', str(x)).group(1)) if re.search(r'round(\d+)', str(x)) else 0)[-1])
                except:
                    # If no round number, use most recently modified
                    rec_model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))
                args.rec_model_path = rec_model_path
                logging.info(f"Found model for evaluation: {rec_model_path}")
        
        if rec_model_path and os.path.exists(rec_model_path):
            # Run evaluation
            args.train = 0  # Set to test mode
            logging.info(f"Running evaluation with model: {rec_model_path}")
            single_main()
        else:
            logging.warning(f"Model not found at {model_path}, will try to extract metrics from training log")
            # Metrics might be computed during training if validation includes metrics
        
        # Extract metrics from log file
        metrics_list = [m.strip().lower() for m in args.metrics.split(',')]
        metrics_dict = extract_metrics_from_log(test_log_file, metrics_list)
        
        if metrics_dict is None:
            # Try train log as fallback
            metrics_dict = extract_metrics_from_log(train_log_file, metrics_list)
        
        if metrics_dict:
            # Get the metrics to optimize (support both single and multi-objective)
            optimize_metrics = args.optimize_metrics.split(',') if args.optimize_metrics else []
            # Backward compatibility: if optimize_metric is set, use it
            if args.optimize_metric:
                optimize_metrics = [args.optimize_metric]
            
            optimize_metrics = [m.strip().lower() for m in optimize_metrics]
            
            # Extract metric values
            metric_values = []
            missing_metrics = []
            for metric in optimize_metrics:
                if metric in metrics_dict:
                    metric_values.append(metrics_dict[metric])
                else:
                    missing_metrics.append(metric)
            
            if missing_metrics:
                logging.warning(f"Metrics {missing_metrics} not found. Available: {list(metrics_dict.keys())}")
            
            if metric_values:
                if len(metric_values) == 1:
                    # Single objective
                    logging.info(f"Trial {trial.number} completed with {optimize_metrics[0]}={metric_values[0]:.4f}")
                    return metric_values[0]
                else:
                    # Multi-objective: return tuple
                    metric_str = ", ".join([f"{m}={v:.4f}" for m, v in zip(optimize_metrics[:len(metric_values)], metric_values)])
                    logging.info(f"Trial {trial.number} completed with metrics: {metric_str}")
                    return tuple(metric_values)
            else:
                # Fallback to first available metric
                if metrics_dict:
                    first_metric = list(metrics_dict.keys())[0]
                    metric_value = metrics_dict[first_metric]
                    logging.info(f"Using fallback metric {first_metric}={metric_value:.4f}")
                    return metric_value
        
        # Fallback: try to extract validation loss
        logging.warning(f"Could not extract metrics from logs, trying validation loss as fallback")
        best_val_loss = None
        if train_log_file.exists():
            with open(train_log_file, 'r') as f:
                for line in f:
                    match = re.search(r'Validation Loss[:\s]+([\d.]+)', line)
                    if match:
                        val_loss = float(match.group(1))
                        if best_val_loss is None or val_loss < best_val_loss:
                            best_val_loss = val_loss
        
        if best_val_loss is not None:
            logging.info(f"Trial {trial.number} completed with validation loss: {best_val_loss:.4f}")
            return best_val_loss
        
        # Return a poor value if nothing found
        logging.error(f"Could not extract any metric or loss from logs")
        # Determine if multi-objective
        optimize_metrics = args.optimize_metrics.split(',') if args.optimize_metrics else []
        if args.optimize_metric:
            optimize_metrics = [args.optimize_metric]
        optimize_metrics = [m.strip().lower() for m in optimize_metrics]
        
        if len(optimize_metrics) > 1:
            # Multi-objective: return tuple of poor values
            return tuple([float('-inf')] * len(optimize_metrics))
        else:
            # Single objective
            return float('-inf') if args.direction == 'maximize' else float('inf')
            
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # Return a poor value so Optuna knows this trial failed
        optimize_metrics = args.optimize_metrics.split(',') if args.optimize_metrics else []
        if args.optimize_metric:
            optimize_metrics = [args.optimize_metric]
        optimize_metrics = [m.strip().lower() for m in optimize_metrics]
        
        if len(optimize_metrics) > 1:
            return tuple([float('-inf')] * len(optimize_metrics))
        else:
            return float('-inf') if args.direction == 'maximize' else float('inf')
    finally:
        # Restore original values
        args.run_id = original_run_id
        args.train = original_train

def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for lambda_mask and lambda_kl')
    parser = utils.parse_global_args(parser)
    
    # Optuna-specific arguments
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default="lambda_tuning", help="Optuna study name")
    parser.add_argument("--direction", type=str, default="maximize", choices=["minimize", "maximize"],
                       help="Direction to optimize (maximize for metrics like NDCG/Hit, minimize for loss). Ignored if --optimize_metrics is used.")
    parser.add_argument("--optimize_metric", type=str, default=None,
                       help="[DEPRECATED] Single metric to optimize. Use --optimize_metrics instead.")
    parser.add_argument("--optimize_metrics", type=str, default="ndcg@5,ndcg@10,hit@5,hit@10",
                       help="Comma-separated list of metrics to optimize (e.g., 'ndcg@5,ndcg@10,hit@5,hit@10'). Must be in --metrics list. For multi-objective optimization.")
    parser.add_argument("--storage", type=str, default=None, 
                       help="Optuna storage URL (e.g., 'sqlite:///optuna.db' for persistent storage)")
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "nop", "percentile"],
                       help="Optuna pruner type")
    
    args, extras = parser.parse_known_args()
    
    # Ensure diffusion is enabled (required for lambda parameters)
    if not args.use_diffusion:
        logging.warning("use_diffusion is 0, but lambda_mask and lambda_kl require diffusion. Setting use_diffusion=1")
        args.use_diffusion = 1
    
    # Setup logging
    utils.setup_logging(args)
    
    # Validate optimize_metrics are in metrics list
    metrics_list = [m.strip().lower() for m in args.metrics.split(',')]
    
    # Determine which metrics to optimize
    optimize_metrics = []
    if args.optimize_metrics:
        optimize_metrics = [m.strip().lower() for m in args.optimize_metrics.split(',')]
    elif args.optimize_metric:
        # Backward compatibility
        optimize_metrics = [args.optimize_metric.strip().lower()]
    else:
        # Default: optimize all NDCG and Hit@5 and @10
        default_metrics = ['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10']
        optimize_metrics = [m for m in default_metrics if m in metrics_list]
        if not optimize_metrics:
            # Fallback to first available metric
            optimize_metrics = [metrics_list[0]] if metrics_list else []
    
    # Validate all metrics are in the metrics list
    invalid_metrics = [m for m in optimize_metrics if m not in metrics_list]
    if invalid_metrics:
        logging.warning(f"Metrics {invalid_metrics} not in metrics list: {metrics_list}")
        optimize_metrics = [m for m in optimize_metrics if m in metrics_list]
    
    if not optimize_metrics:
        logging.error("No valid metrics to optimize!")
        sys.exit(1)
    
    # Determine optimization direction(s)
    is_multi_objective = len(optimize_metrics) > 1
    if is_multi_objective:
        # Multi-objective: all metrics should be maximized (for NDCG/Hit)
        directions = ['maximize'] * len(optimize_metrics)
    else:
        # Single objective: use user-specified direction or default to maximize
        directions = [args.direction] if args.direction else ['maximize']
    
    logging.info("="*80)
    logging.info("Starting Optuna hyperparameter tuning")
    logging.info(f"Study name: {args.study_name}")
    logging.info(f"Number of trials: {args.n_trials}")
    if is_multi_objective:
        logging.info(f"Multi-objective optimization: {len(optimize_metrics)} metrics")
        for i, metric in enumerate(optimize_metrics):
            logging.info(f"  Objective {i+1}: {metric} ({directions[i]})")
    else:
        logging.info(f"Single-objective optimization: {optimize_metrics[0]} ({directions[0]})")
    logging.info(f"Tuning parameters: lambda_mask, lambda_kl")
    logging.info("="*80)
    
    # Store optimize_metrics in args for use in objective function
    args.optimize_metrics = ','.join(optimize_metrics)
    
    # Create or load study (multi-objective if multiple metrics)
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            directions=directions,
            storage=args.storage,
            load_if_exists=True,
            pruner=get_pruner(args.pruner)
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            directions=directions,
            pruner=get_pruner(args.pruner)
        )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    logging.info("="*80)
    logging.info("Optuna tuning completed!")
    
    if is_multi_objective:
        # Multi-objective: show Pareto front
        logging.info(f"Number of Pareto-optimal trials: {len(study.best_trials)}")
        logging.info("\nPareto-optimal trials (best trade-offs):")
        for i, trial in enumerate(study.best_trials):
            logging.info(f"\nPareto solution {i+1} (Trial {trial.number}):")
            if isinstance(trial.values, (list, tuple)):
                for j, metric in enumerate(optimize_metrics):
                    if j < len(trial.values):
                        logging.info(f"  {metric}: {trial.values[j]:.4f}")
            logging.info(f"  Parameters:")
            for key, value in trial.params.items():
                logging.info(f"    {key}: {value:.4f}")
        
        # Also show the trial with best combined score (sum of normalized metrics)
        if study.best_trials:
            logging.info("\nAll Pareto-optimal trials summary:")
            for trial in study.best_trials:
                values_str = ", ".join([f"{v:.4f}" for v in trial.values]) if isinstance(trial.values, (list, tuple)) else str(trial.values)
                logging.info(f"Trial {trial.number}: values=[{values_str}], "
                           f"lambda_mask={trial.params.get('lambda_mask', 'N/A'):.4f}, "
                           f"lambda_kl={trial.params.get('lambda_kl', 'N/A'):.4f}")
    else:
        # Single objective
        logging.info(f"Best trial: {study.best_trial.number}")
        if isinstance(study.best_value, (list, tuple)):
            values_str = ", ".join([f"{v:.4f}" for v in study.best_value])
            logging.info(f"Best values: [{values_str}]")
        else:
            logging.info(f"Best value: {study.best_value:.4f}")
        logging.info(f"Best parameters:")
        for key, value in study.best_params.items():
            logging.info(f"  {key}: {value:.4f}")
    
    logging.info("="*80)
    
    # Print all trials summary
    logging.info("\nAll trials summary:")
    for trial in study.trials:
        if isinstance(trial.value, (list, tuple)):
            values_str = ", ".join([f"{v:.4f}" for v in trial.value])
            logging.info(f"Trial {trial.number}: values=[{values_str}], "
                        f"lambda_mask={trial.params.get('lambda_mask', 'N/A'):.4f}, "
                        f"lambda_kl={trial.params.get('lambda_kl', 'N/A'):.4f}")
        else:
            logging.info(f"Trial {trial.number}: value={trial.value:.4f}, "
                        f"lambda_mask={trial.params.get('lambda_mask', 'N/A'):.4f}, "
                        f"lambda_kl={trial.params.get('lambda_kl', 'N/A'):.4f}")

def get_pruner(pruner_name):
    """Get Optuna pruner by name"""
    if pruner_name == "median":
        return optuna.pruners.MedianPruner()
    elif pruner_name == "percentile":
        return optuna.pruners.PercentilePruner(percentile=25.0)
    elif pruner_name == "nop":
        return optuna.pruners.NopPruner()
    else:
        return optuna.pruners.MedianPruner()

if __name__ == "__main__":
    main()

