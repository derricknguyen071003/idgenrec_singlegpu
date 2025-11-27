import optuna
import torch
import logging
import argparse
import os
import sys
from typing import Dict, Any
import wandb
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
from data.MultiTaskDataset_gen import MultiTaskDatasetGen
from data.MultiTaskDataset_rec import MultiTaskDatasetRec
from runner.SingleRunner import SingleRunner
from utils.dataset_utils import get_dataset_generative, get_loader
from undecorated import undecorated
from types import MethodType
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils
from utils.discrete_diffusion import prepend_timestep_token


def objective(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    hyperparams = suggest_hyperparameters(trial, base_args)
    for key, value in hyperparams.items():
        setattr(base_args, key, value)
    
    logging.info(f"Trial {trial.number}: Hyperparameters = {hyperparams}")
    
    if base_args.use_wandb:
        wandb.init(
            project=base_args.run_id if hasattr(base_args, 'run_id') else "optuna_tune",
            name=f"trial_{trial.number}",
            config=hyperparams,
            reinit=True
        )
    
    try:
        original_run_id = base_args.run_id
        base_args.run_id = f"{original_run_id}_trial_{trial.number}"
        validation_metric = run_training_with_validation(trial, base_args)
        return validation_metric
    except Exception as e:
        logging.error(f"Trial {trial.number} failed with error: {e}")
        return 0.0
    finally:
        base_args.run_id = original_run_id
        if base_args.use_wandb:
            wandb.finish()


def suggest_hyperparameters(trial: optuna.Trial, base_args: argparse.Namespace) -> Dict[str, Any]:
    hyperparams = {}
    hyperparams['id_lr'] = trial.suggest_float('id_lr', 1e-5, 1e-2, log=True)
    hyperparams['rec_lr'] = trial.suggest_float('rec_lr', 1e-6, 1e-3, log=True)
    hyperparams['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    hyperparams['warmup_prop'] = trial.suggest_float('warmup_prop', 0.01, 0.2, step=0.01)
    hyperparams['gradient_accumulation_steps'] = trial.suggest_int('gradient_accumulation_steps', 1, 8)
    hyperparams['clip'] = trial.suggest_float('clip', 0.1, 5.0, step=0.1)
    if base_args.use_diffusion:
        hyperparams['lambda_mask'] = trial.suggest_float('lambda_mask', 0.01, 1.0, log=True)
        hyperparams['lambda_kl'] = trial.suggest_float('lambda_kl', 0.01, 1.0, log=True)
        hyperparams['diffusion_beta_max'] = trial.suggest_float('diffusion_beta_max', 0.01, 0.5, step=0.01)
        hyperparams['diffusion_cross_prob'] = trial.suggest_float('diffusion_cross_prob', 0.1, 0.9, step=0.1)
        hyperparams['noise_head_dropout'] = trial.suggest_float('noise_head_dropout', 0.0, 0.5, step=0.05)
    return hyperparams


def run_training_with_validation(trial: optuna.Trial, args: argparse.Namespace) -> float:
    
    device = torch.device(f"cuda:{args.item_gpu}")
    utils.set_seed(args.seed)
    
    config = T5Config.from_pretrained(args.backbone)
    model_gen_item = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
    model_gen_item.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model_rec = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
    
    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    generate_with_grad = undecorated(model_gen_item.generate)
    model_gen_item.generate_with_grad = MethodType(generate_with_grad, model_gen_item)
    model_rec.to(device)
    model_rec.resize_token_embeddings(len(tokenizer))
    model_gen_item.resize_token_embeddings(len(tokenizer))
    
    original_rounds = args.rounds
    args.rounds = 1
    
    try:
        if args.run_type == '2id2rec' or args.run_type == '2id2rec_socialtoid':
            # For 2id2rec, need two runners (item and friend)
            model_gen_friend = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
            model_gen_friend.to(device)
            generate_with_grad = undecorated(model_gen_friend.generate)
            model_gen_friend.generate_with_grad = MethodType(generate_with_grad, model_gen_friend)
            model_gen_friend.resize_token_embeddings(len(tokenizer))
            
            model_social = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
            model_social.to(device)
            generate_with_grad = undecorated(model_social.generate)
            model_social.generate_with_grad = MethodType(generate_with_grad, model_social)
            model_social.resize_token_embeddings(len(tokenizer))
            
            TrainSetID_item, TrainSetRec_item = get_dataset_generative(
                args, model_gen_item, tokenizer, regenerate=False, component='item_rec'
            )
            train_loader_id_item, train_loader_rec_item = get_loader(args, tokenizer, TrainSetID_item, TrainSetRec_item)
            
            TrainSetID_friend, TrainSetRec_friend = get_dataset_generative(
                args, model_gen_friend, tokenizer, regenerate=False, component='friend_rec'
            )
            train_loader_id_friend, train_loader_rec_friend = get_loader(args, tokenizer, TrainSetID_friend, TrainSetRec_friend)
            
            runner_item = SingleRunner(
                model_gen=model_gen_item,
                model_rec=model_rec,
                tokenizer=tokenizer,
                train_loader_id=train_loader_id_item,
                train_loader_rec=train_loader_rec_item,
                device=device,
                args=args,
                component='item_rec',
                other_view_model=model_social
            )
            runner_friend = SingleRunner(
                model_gen=model_gen_friend,
                model_rec=model_social,
                tokenizer=tokenizer,
                train_loader_id=train_loader_id_friend,
                train_loader_rec=train_loader_rec_friend,
                device=device,
                args=args,
                component='friend_rec',
                other_view_model=model_rec
            )
            
            runner_friend._train_id_generator_phase(0)
            runner_friend._train_recommender_phase(0)
            runner_item._train_id_generator_phase(0)
            runner_item._train_recommender_phase(0)
            val_ratio = getattr(args, 'pseudo_val_ratio', 0.15)
            validation_metric = get_training_loss_metric(runner_item, args, validation_ratio=val_ratio)
            del runner_item, runner_friend, model_gen_friend, model_social
            torch.cuda.empty_cache()
            
        else:
            TrainSetID_item, TrainSetRec = get_dataset_generative(
                args, model_gen_item, tokenizer, regenerate=False
            )
            train_loader_id, train_loader_rec = get_loader(args, tokenizer, TrainSetID_item, TrainSetRec)
            runner = SingleRunner(
                model_rec=model_rec,
                model_gen=model_gen_item,
                tokenizer=tokenizer,
                train_loader_id=train_loader_id,
                train_loader_rec=train_loader_rec,
                device=device,
                args=args,
            )
            runner.train()
            val_ratio = getattr(args, 'pseudo_val_ratio', 0.15)
            validation_metric = get_training_loss_metric(runner, args, validation_ratio=val_ratio)
            del runner
            torch.cuda.empty_cache()
        
        return validation_metric
    finally:
        args.rounds = original_rounds
        del model_rec, model_gen_item
        torch.cuda.empty_cache()


def get_training_loss_metric(runner, args, validation_ratio=0.15):
    val_ratio = getattr(args, 'pseudo_val_ratio', 0.15) if validation_ratio == 0.15 else validation_ratio
    logging.info(f"Using held-out training subset ({val_ratio*100:.1f}%) as pseudo-validation.")
    runner.model_rec.eval()
    losses = []
    total_batches = len(runner.train_loader_rec)
    num_val_batches = max(1, int(total_batches * val_ratio))
    num_val_batches = min(num_val_batches, 50)
    
    import random
    random.seed(hash(str(args.run_id)) % (2**32))
    batch_indices = random.sample(range(total_batches), min(num_val_batches, total_batches))
    batch_indices.sort()
    
    with torch.no_grad():
        for i, batch in enumerate(runner.train_loader_rec):
            if i not in batch_indices or len(losses) >= num_val_batches:
                continue
            
            input_ids = batch[0].to(runner.device, non_blocking=True)
            attn_mask = batch[1].to(runner.device, non_blocking=True)
            output_ids = batch[3].to(runner.device, non_blocking=True)
            
            if runner.use_diffusion and len(batch) > 5:
                timesteps = batch[5].to(runner.device, non_blocking=True)
                noise_masks = batch[6].to(runner.device, non_blocking=True)
                input_ids, attn_mask = prepend_timestep_token(
                    input_ids, timesteps, runner.timestep_token_ids, attn_mask
                )
            
            token_embeddings = runner.model_rec.shared.weight
            input_embeds = token_embeddings[input_ids]
            output = runner.model_rec(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                labels=output_ids,
                return_dict=True,
            )
            losses.append(output.loss.item())
    
    if not losses:
        logging.warning("No validation batches evaluated. Using fallback metric.")
        return 0.0
    
    avg_loss = sum(losses) / len(losses)
    logging.info(f"Pseudo-validation loss (from {len(losses)}/{total_batches} batches): {avg_loss:.4f}")
    return -avg_loss


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Tuning for IDGenRec')
    parser = utils.parse_global_args(parser)
    
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='idgenrec_optuna', help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--direction', type=str, default='maximize', choices=['maximize', 'minimize'],
                       help='Direction of optimization')
    parser.add_argument('--pruner', type=str, default='median', choices=['median', 'nop', 'successive_halving'],
                       help='Pruner type')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for optimization')
    parser.add_argument('--pseudo_val_ratio', type=float, default=0.15,
                       help='Fraction of training batches to use as pseudo-validation (default: 0.15)')
    
    args, extras = parser.parse_known_args()
    utils.setup_logging(args)
    
    if args.use_wandb:
        utils.setup_wandb(args)
    
    study_kwargs = {'study_name': args.study_name, 'direction': args.direction}
    if args.storage:
        study_kwargs['storage'] = args.storage
        study_kwargs['load_if_exists'] = True
    
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif args.pruner == 'successive_halving':
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    else:
        pruner = optuna.pruners.NopPruner()
    
    study = optuna.create_study(pruner=pruner, **study_kwargs)
    logging.info(f"Starting Optuna optimization with {args.n_trials} trials")
    
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True
    )
    
    logging.info("=" * 80)
    logging.info("Optimization finished!")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    logging.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logging.info(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    if study.best_trial:
        logging.info(f"Best trial: Value: {study.best_trial.value}, Params: {study.best_trial.params}")
    
    if args.storage:
        logging.info(f"Study saved to: {args.storage}")
    else:
        import pickle
        study_file = f"optuna_study_{args.study_name}.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        logging.info(f"Study saved to: {study_file}")
    
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.write_html(f"optuna_history_{args.study_name}.html")
        logging.info(f"Optimization history saved to: optuna_history_{args.study_name}.html")
    except ImportError:
        logging.warning("Plotly not available. Skipping visualization.")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

