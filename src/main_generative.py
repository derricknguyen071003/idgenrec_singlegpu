import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
torch.cuda.empty_cache()
import os
import argparse
import logging
import datetime

from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM 
from data.MultiTaskDataset_gen import MultiTaskDatasetGen # For TrainSetID
from data.MultiTaskDataset_rec import MultiTaskDatasetRec # For TrainSetRec
from runner.SingleRunner import SingleRunner 

from processor.Collator import CollatorGen, Collator, TestCollator
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler

from utils import utils
from utils import initialization # If used for model init
from utils.dataset_utils import get_dataset_generative, get_loader
from undecorated import undecorated
from types import MethodType



def single_main():
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='IDGenRec Single GPU Alternating Training')
    parser = utils.parse_global_args(parser) 
    MultiTaskDatasetGen.parse_dataset_args(parser) 
    SingleMultiDataTaskSampler.parse_sampler_args(parser) 
    parser = SingleRunner.parse_runner_args(parser) 
    args, extras = parser.parse_known_args()

    # Setup logging, seed, device
    utils.set_seed(args.seed)
    utils.setup_logging(args) 
    utils.setup_model_path(args) 

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")
    args.device = device 
    args.distributed = 0 
    args.rank = 0        
    args.world_size = 1  

    logging.info("========= Initializing for Single GPU Alternating Training =========")
    #logging.info(f"Script Arguments: {vars(args)}")
    logging.info(f"Using device: {device}")
    config = T5Config.from_pretrained(args.backbone)
    model_gen = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
    model_rec = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # generate with gradient
    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    model_rec.to(device)
    generate_with_grad = undecorated(model_gen.generate)
    model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    model_gen.to(device)

    model_rec.resize_token_embeddings(len(tokenizer))
    model_gen.resize_token_embeddings(len(tokenizer))
    logging.info(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    if getattr(args, 'id_model_path', None):
            logging.info(f"LOAD GEN CHECKPOINT for model_gen from: {args.id_model_path}")
            model_gen.load_state_dict(torch.load(args.id_model_path, map_location=device))

    if getattr(args, 'rec_model_path', None):
            logging.info(f"LOAD REC CHECKPOINT for model_rec from: {args.rec_model_path}")
            model_rec.load_state_dict(torch.load(args.rec_model_path, map_location=device))

    #region # 2. Load model_gen (ID Generator Model)
    # id_model_backbone_name = getattr(args, 'id_model_backbone', None) or args.backbone
    # try:
    #     # Load config and modify vocab size to match checkpoint
    #     config = T5Config.from_pretrained(id_model_backbone_name)
        
    #     # Set vocab_size to your desired value
    #     config.vocab_size = 32100  # Change this to your desired vocab size

    #     # Load the model with the modified config
    #     model_gen = T5ForConditionalGeneration.from_pretrained(
    #         "nandakishormpai/t5-small-machine-articles-tag-generation",
    #         config=config,
    #         ignore_mismatched_sizes=True  # Allows loading even if vocab_size differs
    #     )

    #     # Load checkpoint
    #     if getattr(args, 'id_model_path', None):
    #         logging.info(f"LOAD GEN CHECKPOINT for model_gen from: {args.id_model_path}")
    #         model_gen.load_state_dict(torch.load(args.id_model_path, map_location=device))
    #     # generate_with_grad = undecorated(model_gen.generate)
    #     # model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    #     # model_gen.to(device)

    #     logging.info(f"GEN: ({type(model_gen).__name__}) on {device} with vocab size {getattr(model_gen.config, 'vocab_size', 'unknown')}.")
    # except Exception as e:
    #     logging.error(f"Failed to load model_gen from '{id_model_backbone_name}': {e}")
    #     return


    # # Recommender Model (model_rec)
    # rec_model_backbone_name = getattr(args, 'rec_model_backbone', None) or args.backbone
    # try:
    #     # Load config first to allow overriding vocab_size if needed
    #     config_rec = T5Config.from_pretrained(rec_model_backbone_name)
        
    #     # Optional: Set vocab size to match known checkpoint size (e.g., 32100)
    #     config_rec.vocab_size = 32100  # Only if you know this is true

    #     # Initialize the model using that config
    #     model_rec = T5ForConditionalGeneration(config_rec)

    #     # Load checkpoint if provided
    #     if getattr(args, 'rec_model_path', None):
    #         logging.info(f"LOAD REC CHECKPOINT for model_rec from: {args.rec_model_path}")
    #         model_rec.load_state_dict(torch.load(args.rec_model_path, map_location=device))

    #     generate_with_grad = undecorated(model_rec.generate)
    #     model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    #     model_rec.to(device)
    #     model_rec.to(device)
    #     logging.info(f"REC: ({type(model_rec).__name__}) on {device} with vocab size {getattr(model_rec.config, 'vocab_size', 'unknown')}.")
    # except Exception as e:
    #     logging.error(f"Failed to load model_rec from '{rec_model_backbone_name}': {e}")
    #     return

    #endregion logging.info("="*40)

    # 3. Prepare Datasets (TrainSetID, TrainSetRec, ValidSet)
    TrainSetID, TrainSetRec, ValidSet = get_dataset_generative(args, model_gen, tokenizer)                                                              
                                                               
    logging.info(f"TrainSetID type: {type(TrainSetID)}, length: {len(TrainSetID) if TrainSetID else 'None'}")
    logging.info(f"TrainSetRec type: {type(TrainSetRec)}, length: {len(TrainSetRec) if TrainSetRec else 'None'}")
    #logging.info(f"ValidSet type: {type(ValidSet)}, length: {len(ValidSet) if ValidSet else 'None'}")
   
    # output_path_id = os.path.join(args.model_path if hasattr(args, 'model_path') else '.', "trainsetid_examples.txt")
    # try:
    #     with open(output_path_id, "w", encoding="utf-8") as f:
    #         for idx, example in enumerate(TrainSetID):
    #             f.write(f"Example {idx}:\n{str(example)}\n\n")
    #     logging.info(f"All TrainSetID examples written to {output_path_id}")
    # except Exception as e:
    #     logging.error(f"Failed to write TrainSetID examples to file: {e}")

    # # Print all TrainSetRec examples into a file
    # output_path_rec = os.path.join(args.model_path if hasattr(args, 'model_path') else '.', "trainsetrec_examples.txt")
    # try:
    #     with open(output_path_rec, "w", encoding="utf-8") as f:
    #         for idx, example in enumerate(TrainSetRec):
    #             f.write(f"Example {idx}:\n{str(example)}\n\n")
    #     logging.info(f"All TrainSetRec examples written to {output_path_rec}")
    # except Exception as e:
    #     logging.error(f"Failed to write TrainSetRec examples to file: {e}")

    logging.info("*"*40)
   

    # 4. Prepare DataLoaders
    train_loader_id, train_loader_rec, valid_loader = get_loader(args, tokenizer, TrainSetID, TrainSetRec, ValidSet)
    if not train_loader_id or not train_loader_rec:
        logging.error("Failed to create necessary training data loaders for alternating training. Aborting.")
        if not train_loader_id: logging.error("train_loader_id is None.")
        if not train_loader_rec: logging.error("train_loader_rec is None.")
        return

    # 5. Initialize SingleRunner
    logging.info("INITIALIZE SingleRunner...")
    try:
        runner = SingleRunner(
            model_rec=model_rec,
            model_gen=model_gen,
            tokenizer=tokenizer,
            train_loader_id=train_loader_id,
            train_loader_rec=train_loader_rec,
            valid_loader=valid_loader, 
            device=device,
            args=args
        )
        logging.info("INITIALIZE SINGLE RUNNER SUCCESSFULLY.")
        logging.info("="*40)
    except Exception as e:
        logging.error(f"Error initializing SingleRunner: {e}")
        logging.exception("Detailed traceback for SingleRunner init:")
        return

    # 6. Start Training
    if args.train:
        logging.info(f"Starting training with alt_style: '{args.alt_style}', rounds: {args.rounds}")
        try:
            runner.train() 
            logging.info("Training finished.")
        except Exception as e:
            logging.error(f"Error during runner.train(): {e}")
            logging.exception("Detailed traceback for runner.train():")
    else:
        logging.info("args.train is 0, skipping training. Running test if enabled...")
        runner._test_both_models()

    logging.info("========= single_main finished =========")
    endtime = datetime.datetime.now()
    logging.info(f"Total time taken: {endtime - start_time}")

if __name__ == "__main__":
    single_main()