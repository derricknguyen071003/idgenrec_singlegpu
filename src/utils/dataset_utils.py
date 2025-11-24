from data.MultiTaskDataset_gen import MultiTaskDatasetGen
from data.MultiTaskDataset_rec import MultiTaskDatasetRec
from data.MultiTaskDataset_social import MultiTaskDatasetSocial
from data.MultiTaskDataset_rec_social import MultiTaskDatasetRecSocial
from data.TestDataset import TestDatasetGen
from data.TestDataset_social import TestDatasetSocial
from torch.utils.data import ConcatDataset, DataLoader
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
from processor.Collator import CollatorGen, Collator
import logging

def get_dataset_generative(args, model_gen, tokenizer, phase=0, regenerate=True, component=None):
    datasets = args.datasets.split(',')
    train_all_datasets_id = []
    train_all_datasets_rec = []
    if args.run_type == 'original_idgenrec':
        logging.info(f"Running original idgenrec")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRec)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        logging.info(TrainSetID[0])
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        logging.info(TrainSetRec[0])
        return TrainSetID, TrainSetRec
    elif args.run_type == 'social_to_rec':
        logging.info(f"Running social to rec")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRecSocial)
            train_all_datasets_rec.append(TrainDatasetRec)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        return TrainSetID, TrainSetRec
    elif args.run_type == 'social_to_id':
        logging.info(f"Running social to id")
        TrainSetSocial = []
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer)
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_id.append(TrainDatasetSocial)
            train_all_datasets_rec.append(TrainDatasetRec)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        return TrainSetID, TrainSetRec
    elif args.run_type == 'social_to_both':
        logging.info(f"Running social to rec and id")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer)
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_id.append(TrainDatasetSocial)
            train_all_datasets_rec.append(TrainDatasetRecSocial)
            train_all_datasets_rec.append(TrainDatasetRec)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        return TrainSetID, TrainSetRec
    elif args.run_type == '1id2rec':
        logging.info("Train IDGenerator on both item rec and friend rec")
        train_all_datasets_social = []
        train_all_datasets_rec_social = []
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)  
            TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer)
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_social.append(TrainDatasetSocial)
            train_all_datasets_rec_social.append(TrainDatasetRecSocial)
            train_all_datasets_rec.append(TrainDatasetRec)        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        TrainSetRecSocial = ConcatDataset(train_all_datasets_rec_social)    
        logging.info(f"TrainSetRecSocial: {len(TrainSetRecSocial)}")    
        TrainSetSocial = ConcatDataset(train_all_datasets_social)
        logging.info(f"TrainSetSocial: {len(TrainSetSocial)}")
        logging.info(f"TrainSetID (idgenrec social): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (idgenrec social): {len(TrainSetRec)}")
        logging.info(f"TrainSetRecSocial (idgenrec social): {len(TrainSetRecSocial)}")
        logging.info(f"TrainSetSocial (idgenrec social): {len(TrainSetSocial)}")
        return TrainSetID, TrainSetRec, TrainSetRecSocial, TrainSetSocial
    elif args.run_type == '2id2rec':
        if component == 'item_rec':
            logging.info("Train IDGenerator on item rec")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='item_rec')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='item_rec')  
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (idgenrec itemrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (idgenrec itemrec): {len(TrainSetRec)}")
            return TrainSetID, TrainSetRec
        elif component == 'friend_rec':
            logging.info("Train IDGenerator on friend rec")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
                TrainDatasetID = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')  
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (idgenrec friendrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (idgenrec friendrec): {len(TrainSetRec)}")
            return TrainSetID, TrainSetRec
        else:
            raise ValueError(f"run_type '2id2rec' requires component to be 'item_rec' or 'friend_rec', got: {component}")
    elif args.run_type == '2id2rec_socialtoid':
        if component == 'item_rec':
            logging.info("Train IDGenerator on item rec (item+social to id)")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            train_all_datasets_social = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='item_rec')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='item_rec')  
                TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
                train_all_datasets_id.append(TrainDatasetSocial)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (idgenrec itemrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (idgenrec itemrec): {len(TrainSetRec)}")
            return TrainSetID, TrainSetRec
        elif component == 'friend_rec':
            logging.info("Train IDGenerator on friend rec")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            train_all_datasets_social = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')  
                TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_id.append(TrainDatasetSocial)
                train_all_datasets_rec.append(TrainDatasetRec)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (idgenrec friendrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (idgenrec friendrec): {len(TrainSetRec)}")
            return TrainSetID, TrainSetRec
        else:
            raise ValueError(f"run_type '2id2rec_socialtoid' requires component to be 'item_rec' or 'friend_rec', got: {component}")
    elif args.run_type == '2id1rec':
        if component == 'item_view':
            logging.info("Train IDGenerator on item view")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='item_view')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='item_view')  
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (item view): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (item view): {len(TrainSetRec)}")
            return TrainSetID, TrainSetRec
        elif component == 'social_view':
            logging.info("Train IDGenerator on social view")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='social_view')
                TrainDatasetID = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='social_view')  
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (social view): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (social view): {len(TrainSetRec)}")
            return TrainSetID, TrainSetRec
        else:
            raise ValueError(f"run_type '2id1rec' requires component to be 'item_view' or 'social_view', got: {component}")
    elif args.run_type == 'idgenrec_friend':
        logging.info("Train IDGenerator on friend rec")
        train_all_datasets_id = []
        train_all_datasets_rec = []
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
            TrainDatasetID = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')  
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRec)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetID (idgenrec friend): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (idgenrec friend): {len(TrainSetRec)}")
        return TrainSetID, TrainSetRec
    elif args.run_type == 'item_to_id_friendrec':
        logging.info(f"Running item to id(friendrec)")
        TrainSetSocial = []
        for data in datasets:
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer)
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_id.append(TrainDatasetSocial)
            train_all_datasets_rec.append(TrainDatasetRecSocial)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetID (item to id friendrec): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (item to id friendrec): {len(TrainSetRec)}")
        return TrainSetID, TrainSetRec
    elif args.run_type == 'item_to_rec_friendrec':
        logging.info(f"Running item to rec(friendrec)")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRec)
            train_all_datasets_rec.append(TrainDatasetRecSocial)
        TrainSetID = ConcatDataset(train_all_datasets_id)
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetID (item to rec friendrec): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (item to rec friendrec): {len(TrainSetRec)}")
        return TrainSetID, TrainSetRec
    else:
        raise ValueError(f"Unsupported run_type: {args.run_type}. Supported types: 'original_idgenrec', 'social_to_rec', 'social_to_id', 'social_to_both', '1id2rec', '2id2rec', '2id2rec_socialtoid', '2id1rec', 'idgenrec_friend', 'item_to_id_friendrec', 'item_to_rec_friendrec'")

def get_loader(args, tokenizer, TrainSetID, TrainSetRec, TrainSetRecSocial=None, TrainSetSocial=None):
    collator_gen = CollatorGen(tokenizer)
    collator_rec = Collator(tokenizer, args=args)

    train_loader_id = None
    train_loader_rec = None

    num_workers = getattr(args, 'num_workers', 32)
    prefetch_factor = 32 if num_workers > 0 else None  
    
    if TrainSetID is not None:
        train_sampler_id = SingleMultiDataTaskSampler(TrainSetID, args.id_batch_size, args.seed, shuffle=True)
        train_loader_id = DataLoader(dataset=TrainSetID, sampler=train_sampler_id, batch_size=args.id_batch_size, collate_fn=collator_gen, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        
        # Show tokenized sequence for TrainSetID example
        # if len(TrainSetID) > 0:
        #     sample_id = TrainSetID[0]
        #     logging.info("=" * 80)
        #     logging.info("TrainSetID Sample - Tokenized Sequence:")
        #     logging.info(f"Original sample: {sample_id}")
        #     batch_id = collator_gen([sample_id])
        #     input_prompt_ids, input_insert_pos, history_ids, history_attention, output_ids, output_attention = batch_id
        #     input_tokens = tokenizer.convert_ids_to_tokens(input_prompt_ids[0].tolist())
        #     output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0].tolist())
        #     logging.info(f"Input prompt tokens: {input_tokens}")
        #     logging.info(f"Input prompt token IDs: {input_prompt_ids[0].tolist()}")
        #     logging.info(f"Output prompt tokens: {output_tokens}")
        #     logging.info(f"Output prompt token IDs: {output_ids[0].tolist()}")
        #     if history_ids.shape[1] > 0 and history_ids.shape[2] > 0:
        #         history_tokens = tokenizer.convert_ids_to_tokens(history_ids[0][0].tolist())
        #         logging.info(f"History tokens (first item): {history_tokens}")
        #     logging.info("=" * 80)
    
    if TrainSetRec is not None:
        train_sampler_rec = SingleMultiDataTaskSampler(TrainSetRec, args.rec_batch_size, args.seed, shuffle=True) 
        train_loader_rec = DataLoader(dataset=TrainSetRec, sampler=train_sampler_rec, batch_size=args.rec_batch_size, collate_fn=collator_rec, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        
        # Show tokenized sequence for TrainSetRec example
        # if len(TrainSetRec) > 0:
        #     sample_rec = TrainSetRec[0]
        #     logging.info("=" * 80)
        #     logging.info("TrainSetRec Sample - Tokenized Sequence:")
        #     logging.info(f"Original sample: {sample_rec}")
        #     batch_rec = collator_rec([sample_rec])
        #     if len(batch_rec) > 5:
        #         input_ids, input_attention, whole_word_ids, output_ids, output_attention, timesteps, noise_masks = batch_rec
        #     else:
        #         input_ids, input_attention, whole_word_ids, output_ids, output_attention = batch_rec
        #     input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        #     output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0].tolist())
        #     logging.info(f"Input tokens: {input_tokens}")
        #     logging.info(f"Input token IDs: {input_ids[0].tolist()}")
        #     logging.info(f"Output tokens: {output_tokens}")
        #     logging.info(f"Output token IDs: {output_ids[0].tolist()}")
        #     logging.info("=" * 80)
    
    if args.run_type == '1id2rec' and TrainSetSocial is not None and TrainSetRecSocial is not None:
        collator_social = CollatorGen(tokenizer)
        collator_rec_social = Collator(tokenizer, args=args)
        train_sampler_social = SingleMultiDataTaskSampler(TrainSetSocial, args.social_batch_size, args.seed, shuffle=True)
        train_sampler_rec_social = SingleMultiDataTaskSampler(TrainSetRecSocial, args.rec_batch_size, args.seed, shuffle=True)
        train_loader_social = DataLoader(dataset=TrainSetSocial, sampler=train_sampler_social, batch_size=args.social_batch_size, collate_fn=collator_social, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        train_loader_rec_social = DataLoader(dataset=TrainSetRecSocial, sampler=train_sampler_rec_social, batch_size=args.rec_batch_size, collate_fn=collator_rec_social, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        return train_loader_id, train_loader_rec, train_loader_rec_social, train_loader_social

    if (args.run_type == '2id2rec' or args.run_type == '2id2rec_socialtoid') and TrainSetSocial is not None and TrainSetRecSocial is not None:
        collator_social = CollatorGen(tokenizer)
        collator_rec_social = Collator(tokenizer, args=args)
        train_sampler_social = SingleMultiDataTaskSampler(TrainSetSocial, args.social_batch_size, args.seed, shuffle=True)
        train_sampler_rec_social = SingleMultiDataTaskSampler(TrainSetRecSocial, args.rec_batch_size, args.seed, shuffle=True)
        train_loader_social = DataLoader(dataset=TrainSetSocial, sampler=train_sampler_social, batch_size=args.social_batch_size, collate_fn=collator_social, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        train_loader_rec_social = DataLoader(dataset=TrainSetRecSocial, sampler=train_sampler_rec_social, batch_size=args.rec_batch_size, collate_fn=collator_rec_social, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        return train_loader_id, train_loader_rec, train_loader_rec_social, train_loader_social
    
    if TrainSetRecSocial is not None and TrainSetID is None and TrainSetRec is None:
        collator_rec_social = Collator(tokenizer, args=args)
        train_sampler_rec_social = SingleMultiDataTaskSampler(TrainSetRecSocial, args.rec_batch_size, args.seed, shuffle=True)
        train_loader_rec_social = DataLoader(dataset=TrainSetRecSocial, sampler=train_sampler_rec_social, batch_size=args.rec_batch_size, collate_fn=collator_rec_social, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=num_workers > 0)
        return train_loader_id, train_loader_rec, train_loader_rec_social, None
    
    return train_loader_id, train_loader_rec
