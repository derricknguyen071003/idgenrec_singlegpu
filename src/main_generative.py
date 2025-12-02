import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress warnings about sequence length exceeding model maximum - we handle truncation explicitly
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than.*")
warnings.filterwarnings("ignore", message=".*sequence length.*longer than.*maximum.*")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
import argparse
import logging
import wandb
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
from runner.SingleRunner import SingleRunner 
from utils import utils
from utils.dataset_utils import get_dataset_generative, get_loader
from undecorated import undecorated
from types import MethodType
from utils import indexing
from utils.indexing import generative_indexing_rec, construct_user_sequence_dict, generate_cross_social_index
def single_main():
    parser = argparse.ArgumentParser(description='IDGenRec Single GPU Alternating Training')
    parser = utils.parse_global_args(parser) 
    args, extras = parser.parse_known_args()
    item_device = torch.device(f"cuda:{args.item_gpu}")
    utils.set_seed(args.seed)
    utils.setup_logging(args) 
    if args.train:
        utils.setup_model_path(args)
        utils.setup_wandb(args)
    logging.info(f"Using device: {item_device}")
    config = T5Config.from_pretrained(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model_gen_item = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
    model_gen_item.to(item_device)  
    model_rec = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)    
    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    generate_with_grad = undecorated(model_gen_item.generate)
    model_gen_item.generate_with_grad = MethodType(generate_with_grad, model_gen_item)
    model_rec.to(item_device)
    model_rec.resize_token_embeddings(len(tokenizer))
    model_gen_item.resize_token_embeddings(len(tokenizer))
    if args.run_type == '2id2rec' or args.run_type == '2id2rec_socialtoid':
        logging.info(f"Running {args.run_type}")
        model_gen_friend = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
        model_gen_friend.to(item_device)
        generate_with_grad = undecorated(model_gen_friend.generate)
        model_gen_friend.generate_with_grad = MethodType(generate_with_grad, model_gen_friend)
        model_gen_friend.resize_token_embeddings(len(tokenizer))

        model_social = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
        model_social.to(item_device)
        generate_with_grad = undecorated(model_social.generate)
        model_social.generate_with_grad = MethodType(generate_with_grad, model_social)
        model_social.resize_token_embeddings(len(tokenizer))

        TrainSetID_item, TrainSetRec, ValSetRec_item = get_dataset_generative(args, model_gen_item, tokenizer, regenerate=False, component='item_rec')
        train_loader_id_item, train_loader_rec_item, val_loader_rec_item = get_loader(args, tokenizer, TrainSetID_item, TrainSetRec, ValSetRec=ValSetRec_item)
        
        TrainSetID_friend, TrainSetRec_friend, ValSetRec_friend = get_dataset_generative(args, model_gen_friend, tokenizer, regenerate=False, component='friend_rec')
        train_loader_id_friend, train_loader_rec_friend, val_loader_rec_friend = get_loader(args, tokenizer, TrainSetID_friend, TrainSetRec_friend, ValSetRec=ValSetRec_friend)
        
        runner_item = SingleRunner(
            model_gen=model_gen_item,
            model_rec=model_rec,
            tokenizer=tokenizer,
            train_loader_id=train_loader_id_item,
            train_loader_rec=train_loader_rec_item,
            device=item_device,
            args=args,
            component = 'item_rec',
            other_view_model=model_social,
            val_loader_rec=val_loader_rec_item
        )
        runner_friend = SingleRunner(
            model_gen=model_gen_friend,
            model_rec=model_social,
            tokenizer=tokenizer,
            train_loader_id=train_loader_id_friend,
            train_loader_rec=train_loader_rec_friend,
            device=item_device,
            args=args,
            component = 'friend_rec',
            other_view_model=model_rec,
            val_loader_rec=val_loader_rec_friend
        )
    else:
        logging.info(f"Running {args.run_type}")
        TrainSetID_item, TrainSetRec, ValSetRec = get_dataset_generative(args, model_gen_item, tokenizer, regenerate=False)
        train_loader_id, train_loader_rec, val_loader_rec = get_loader(args, tokenizer, TrainSetID_item, TrainSetRec, ValSetRec=ValSetRec)
        runner_item = SingleRunner(
            model_rec=model_rec,
            model_gen=model_gen_item,
            tokenizer=tokenizer,
            train_loader_id=train_loader_id,
            train_loader_rec=train_loader_rec,
            device=item_device,
            args=args,
            val_loader_rec=val_loader_rec,
        )
    if args.train:
        if args.run_type == '2id2rec' or args.run_type == '2id2rec_socialtoid':
            logging.info(f"========= Starting {args.run_type} Training =========")
            for round_num in range(args.rounds):
                logging.info(f'Train IDGen (friend) Round {round_num + 1}')
                runner_friend._train_id_generator_phase(round_num)
                logging.info(f'Train Recommender (friend) Round {round_num + 1}')
                runner_friend._train_recommender_phase(round_num)
                friend_phase = runner_friend.total_id_epoch
                if args.social_quantization_id:
                    logging.info(f'Creating initial item user index files for Round {round_num + 1}')
                    user_sequence = utils.ReadLineFromFile(os.path.join(args.data_path, args.datasets, 'user_sequence.txt'))
                    user_sequence_dict = construct_user_sequence_dict(user_sequence)
                    generative_indexing_rec(
                        data_path=args.data_path,
                        dataset=args.datasets,
                        user_sequence_dict=user_sequence_dict,
                        model_gen=model_gen_item,
                        tokenizer=tokenizer,
                        phase=0,
                        regenerate=True,
                        run_id=args.run_id,
                        component='item_rec',
                        run_type=args.run_type
                    )
                    
                    logging.info(f'Generating cross-social index for item component Round {round_num + 1} (using phase {friend_phase} for both social and item)')
                    from utils.indexing import generate_cross_social_index
                    cross_social_map = generate_cross_social_index(
                        data_path=args.data_path,
                        dataset=args.datasets,
                        model_gen=model_gen_friend,
                        tokenizer=tokenizer,
                        social_phase=friend_phase,  # Use friend phase for social index files
                        item_phase=friend_phase,    # Use same phase for item index files
                        run_id=args.run_id,
                        regenerate=True,  # Always regenerate for each round to get fresh cross-social index
                        round_num=round_num,  # Pass round number to create round-specific cross-social index
                    )
                    logging.info(f'Generated cross-social index for {len(cross_social_map)} users')
                
                logging.info(f'Train IDGen (item) Round {round_num + 1}')
                runner_item._train_id_generator_phase(round_num)
                logging.info(f'Train Recommender (item) Round {round_num + 1}')
                runner_item._train_recommender_phase(round_num)

                logging.info(f"========== Finished {args.run_type} Round {round_num + 1}/{args.rounds} ==========")
                if args.model_path: 
                    # Evaluate models before saving (only for final round)
                    if round_num + 1 == args.rounds:
                        logging.info(f"--- Running evaluation before saving models for final round {round_num + 1} ---")
                        # For 2-tower run types: runner_item tests item rec, runner_friend tests friend rec
                        runner_item._test_recommender(use_in_memory=True)
                        runner_friend._test_recommender(use_in_memory=True)
                    
                    os.makedirs(args.model_path, exist_ok=True)
                    logging.info(f"Model directory ensured: {args.model_path}")
                    
                    # Save item models
                    item_gen_path = os.path.join(args.model_path, f"model_gen_item_round{round_num+1}_final.pt")
                    torch.save(runner_item.model_gen.state_dict(), item_gen_path)
                    item_rec_path = os.path.join(args.model_path, f"model_rec_item_round{round_num+1}_final.pt")
                    torch.save(runner_item.model_rec.state_dict(), item_rec_path)
                    
                    # Save friend models
                    friend_gen_path = os.path.join(args.model_path, f"model_gen_friend_round{round_num+1}_final.pt")
                    torch.save(runner_friend.model_gen.state_dict(), friend_gen_path)
                    friend_social_path = os.path.join(args.model_path, f"model_social_friend_round{round_num+1}_final.pt")
                    torch.save(runner_friend.model_rec.state_dict(), friend_social_path)
                    
                    logging.info(f"Saved models for round {round_num + 1}:")
                    logging.info(f"  - Item ID Generator: {item_gen_path}")
                    logging.info(f"  - Item Recommender: {item_rec_path}")
                    logging.info(f"  - Friend ID Generator: {friend_gen_path}")
                    logging.info(f"  - Friend Social Model: {friend_social_path}")
        else:
            logging.info("========= Starting Item Recommender Training =========")
            runner_item.train()
        wandb.finish()
    else:
        logging.info('*'*80)
        if getattr(args, 'social_model_path', None) or getattr(args, 'rec_model_path', None):
            runner_item._test_recommender()
    
if __name__ == "__main__":
    single_main()