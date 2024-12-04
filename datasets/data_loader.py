import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
import samplers.distributed
import numpy as np

def sample_collate(batch):
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)

    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    att_feats = torch.stack(att_feats, 0)  # [B, 3, 384, 384]

    return indices, input_seq, target_seq, gv_feat, att_feats

def sample_collate_val(batch):
    indices, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)

    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    att_feats = torch.stack(att_feats, 0)

    return indices, gv_feat, att_feats


def load_train(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, gv_feat_path, att_feats_folder):
    coco_set = CocoDataset(
        image_ids_path = image_ids_path, 
        input_seq = None, 
        target_seq = None, 
        gv_feat_path = gv_feat_path, 
        att_feats_folder = att_feats_folder,
        seq_per_img = 1, 
        max_feat_num = cfg.DATA_LOADER.MAX_FEAT
    )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader