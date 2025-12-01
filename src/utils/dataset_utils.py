from data.MultiTaskDataset_gen import MultiTaskDatasetGen
from data.MultiTaskDataset_rec import MultiTaskDatasetRec
from data.MultiTaskDataset_social import MultiTaskDatasetSocial
from data.MultiTaskDataset_rec_social import MultiTaskDatasetRecSocial
from torch.utils.data import ConcatDataset, DataLoader
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
from processor.Collator import CollatorGen, Collator
import logging

def get_dataset_generative(args, model_gen, tokenizer, phase=0, regenerate=True, component=None):
    datasets = args.datasets.split(',')
    train_all_datasets_id = []
    train_all_datasets_rec = []
    validation_all_datasets_rec = []
    if args.run_type == 'original_idgenrec':
        logging.info(f"Running original idgenrec")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRec)
            validation_all_datasets_rec.append(ValDatasetRec)
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        logging.info(f"ValidationSetRec: {len(ValSetRec)}")
        return TrainSetID, TrainSetRec, ValSetRec
    elif args.run_type == 'social_to_rec':
        logging.info(f"Running social to rec")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRecSocial)
            train_all_datasets_rec.append(TrainDatasetRec)
            
            # Create validation datasets
            ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            ValDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            validation_all_datasets_rec.append(ValDatasetRecSocial)
            validation_all_datasets_rec.append(ValDatasetRec)
            logging.info(f"Created validation datasets for {data}: {len(ValDatasetRec)} + {len(ValDatasetRecSocial)} samples")
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        if ValSetRec:
            logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
        return TrainSetID, TrainSetRec, ValSetRec
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
            
            # Create validation dataset
            ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            validation_all_datasets_rec.append(ValDatasetRec)
            logging.info(f"Created validation dataset for {data}: {len(ValDatasetRec)} samples")
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        if ValSetRec:
            logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
        return TrainSetID, TrainSetRec, ValSetRec
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
            
            # Create validation datasets
            ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            ValDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            validation_all_datasets_rec.append(ValDatasetRecSocial)
            validation_all_datasets_rec.append(ValDatasetRec)
            logging.info(f"Created validation datasets for {data}: {len(ValDatasetRec)} + {len(ValDatasetRecSocial)} samples")
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        logging.info(f"TrainSetID: {len(TrainSetID)}")
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetRec: {len(TrainSetRec)}")
        
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        if ValSetRec:
            logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
        return TrainSetID, TrainSetRec, ValSetRec
    elif args.run_type == '2id2rec':
        if component == 'item_rec':
            logging.info("Train IDGenerator on item rec")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            validation_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='item_rec')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='item_rec')  
                ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='item_rec')
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
                validation_all_datasets_rec.append(ValDatasetRec)
            
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
            logging.info(f"TrainSetID (idgenrec itemrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (idgenrec itemrec): {len(TrainSetRec)}")
            logging.info(f"ValidationSetRec (itemrec): {len(ValSetRec)}")
            return TrainSetID, TrainSetRec, ValSetRec
        elif component == 'friend_rec':
            logging.info("Train IDGenerator on friend rec")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            validation_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
                TrainDatasetID = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')  
                ValDatasetRec = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='friend_rec')
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
                validation_all_datasets_rec.append(ValDatasetRec)
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
            logging.info(f"TrainSetID (friendrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (friendrec): {len(TrainSetRec)}")
            logging.info(f"ValidationSetRec (friendrec): {len(ValSetRec)}")
            return TrainSetID, TrainSetRec, ValSetRec
        else:
            raise ValueError(f"run_type '2id2rec' requires component to be 'item_rec' or 'friend_rec', got: {component}")
    elif args.run_type == '2id2rec_socialtoid':
        if component == 'item_rec':
            logging.info("Train IDGenerator on item rec (item+social to id)")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            train_all_datasets_social = []
            validation_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='item_rec')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='item_rec')  
                TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
                train_all_datasets_id.append(TrainDatasetSocial)
                ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='item_rec')
                validation_all_datasets_rec.append(ValDatasetRec)
            
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
            logging.info(f"TrainSetID (itemrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (itemrec): {len(TrainSetRec)}")
            logging.info(f"ValidationSetRec (itemrec): {len(ValSetRec)}")
            return TrainSetID, TrainSetRec, ValSetRec
        elif component == 'friend_rec':
            logging.info("Train IDGenerator on friend rec")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            train_all_datasets_social = []
            validation_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
                TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')  
                TrainDatasetSocial = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='friend_rec')
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_id.append(TrainDatasetSocial)
                train_all_datasets_rec.append(TrainDatasetRec)
                ValDatasetRec = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='friend_rec')
                validation_all_datasets_rec.append(ValDatasetRec)
            
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
            logging.info(f"TrainSetID (friendrec): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (friendrec): {len(TrainSetRec)}")
            logging.info(f"ValidationSetRec (friendrec): {len(ValSetRec)}")
            return TrainSetID, TrainSetRec, ValSetRec
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
                
                # Create validation dataset
                ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='item_view')
                validation_all_datasets_rec.append(ValDatasetRec)
                logging.info(f"Created validation dataset for {data}: {len(ValDatasetRec)} samples")
            
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (item view): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (item view): {len(TrainSetRec)}")
            
            ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
            if ValSetRec:
                logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
            return TrainSetID, TrainSetRec, ValSetRec
        elif component == 'social_view':
            logging.info("Train IDGenerator on social view")
            train_all_datasets_id = []
            train_all_datasets_rec = []
            for data in datasets:
                TrainDatasetRec = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='social_view')
                TrainDatasetID = MultiTaskDatasetSocial(args, data, 'train', phase, model_gen, tokenizer, component='social_view')  
                train_all_datasets_id.append(TrainDatasetID)
                train_all_datasets_rec.append(TrainDatasetRec)
                
                # Create validation dataset
                ValDatasetRec = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='social_view')
                validation_all_datasets_rec.append(ValDatasetRec)
                logging.info(f"Created validation dataset for {data}: {len(ValDatasetRec)} samples")
            
            TrainSetID = ConcatDataset(train_all_datasets_id)
            TrainSetRec = ConcatDataset(train_all_datasets_rec)
            logging.info(f"TrainSetID (social view): {len(TrainSetID)}")
            logging.info(f"TrainSetRec (social view): {len(TrainSetRec)}")
            
            ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
            if ValSetRec:
                logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
            return TrainSetID, TrainSetRec, ValSetRec
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
            
            # Create validation dataset
            ValDatasetRec = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='friend_rec')
            validation_all_datasets_rec.append(ValDatasetRec)
            logging.info(f"Created validation dataset for {data}: {len(ValDatasetRec)} samples")
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetID (idgenrec friend): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (idgenrec friend): {len(TrainSetRec)}")
        
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        if ValSetRec:
            logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
        return TrainSetID, TrainSetRec, ValSetRec
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
            
            # Create validation dataset
            ValDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='friend_rec')
            validation_all_datasets_rec.append(ValDatasetRecSocial)
            logging.info(f"Created validation dataset for {data}: {len(ValDatasetRecSocial)} samples")
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetID (item to id friendrec): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (item to id friendrec): {len(TrainSetRec)}")
        
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        if ValSetRec:
            logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
        return TrainSetID, TrainSetRec, ValSetRec
    elif args.run_type == 'item_to_rec_friendrec':
        logging.info(f"Running item to rec(friendrec)")
        for data in datasets:
            TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
            TrainDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate, component='friend_rec')
            TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase, model_gen, tokenizer)        
            train_all_datasets_id.append(TrainDatasetID)
            train_all_datasets_rec.append(TrainDatasetRec)
            train_all_datasets_rec.append(TrainDatasetRecSocial)
            
            # Create validation datasets
            ValDatasetRec = MultiTaskDatasetRec(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False)
            ValDatasetRecSocial = MultiTaskDatasetRecSocial(args, data, 'validation', model_gen, tokenizer, phase, regenerate=False, component='friend_rec')
            validation_all_datasets_rec.append(ValDatasetRec)
            validation_all_datasets_rec.append(ValDatasetRecSocial)
            logging.info(f"Created validation datasets for {data}: {len(ValDatasetRec)} + {len(ValDatasetRecSocial)} samples")
        
        TrainSetID = ConcatDataset(train_all_datasets_id)
        TrainSetRec = ConcatDataset(train_all_datasets_rec)
        logging.info(f"TrainSetID (item to rec friendrec): {len(TrainSetID)}")
        logging.info(f"TrainSetRec (item to rec friendrec): {len(TrainSetRec)}")
        
        ValSetRec = ConcatDataset(validation_all_datasets_rec) if validation_all_datasets_rec else None
        if ValSetRec:
            logging.info(f"ValidationSetRec: {len(ValSetRec)} total samples")
        return TrainSetID, TrainSetRec, ValSetRec
    else:
        raise ValueError(f"Unsupported run_type: {args.run_type}. Supported types: 'original_idgenrec', 'social_to_rec', 'social_to_id', 'social_to_both', '2id2rec', '2id2rec_socialtoid', '2id1rec', 'idgenrec_friend', 'item_to_id_friendrec', 'item_to_rec_friendrec'")

def get_loader(args, tokenizer, TrainSetID, TrainSetRec, TrainSetRecSocial=None, TrainSetSocial=None, ValSetRec=None):
    collator_gen = CollatorGen(tokenizer)
    collator_rec = Collator(tokenizer, args=args)

    train_loader_id = None
    train_loader_rec = None
    val_loader_rec = None
    
    if TrainSetID is not None:
        train_sampler_id = SingleMultiDataTaskSampler(TrainSetID, args.id_batch_size, args.seed, shuffle=True)
        train_loader_id = DataLoader(dataset=TrainSetID, sampler=train_sampler_id, batch_size=args.id_batch_size, collate_fn=collator_gen, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor, persistent_workers=args.num_workers > 0)
    
    if TrainSetRec is not None:
        train_sampler_rec = SingleMultiDataTaskSampler(TrainSetRec, args.rec_batch_size, args.seed, shuffle=True) 
        train_loader_rec = DataLoader(dataset=TrainSetRec, sampler=train_sampler_rec, batch_size=args.rec_batch_size, collate_fn=collator_rec, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor, persistent_workers=args.num_workers > 0)
    
    # Create validation loader if validation dataset is provided
    if ValSetRec is not None:
        val_loader_rec = DataLoader(
            dataset=ValSetRec,
            batch_size=args.rec_batch_size,
            collate_fn=collator_rec,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.num_workers > 0
        )
        logging.info(f"Created validation loader with {len(ValSetRec)} samples")
    
    # Handle case where only TrainSetRecSocial is provided (no TrainSetID or TrainSetRec)
    if TrainSetRecSocial is not None and TrainSetID is None and TrainSetRec is None:
        collator_rec_social = Collator(tokenizer, args=args)
        train_sampler_rec_social = SingleMultiDataTaskSampler(TrainSetRecSocial, args.rec_batch_size, args.seed, shuffle=True)
        train_loader_rec_social = DataLoader(dataset=TrainSetRecSocial, sampler=train_sampler_rec_social, batch_size=args.rec_batch_size, collate_fn=collator_rec_social, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor, persistent_workers=args.num_workers > 0)
        return train_loader_id, train_loader_rec, train_loader_rec_social, None, val_loader_rec
    
    return train_loader_id, train_loader_rec, val_loader_rec
