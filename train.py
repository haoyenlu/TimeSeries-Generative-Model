import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from utils import load_numpy_data, load_config
from dataset import UpperLimbMotionDataset
from model_utils import weight_init, get_trainer_from_config
from train_utils import LinearLrDecay
from models.trainer import cGANTrainer, DiffusionTrainer
from models.generative.GAN import tts_cgan, eeg_cgan
from models.generative.diffusion import diffusion_ts , unet1d
from models.generative.diffusion.transformer import Transformer

from argument import train_argument




def main():
    args = train_argument()
    config = load_config(args.config)

    '''Load Data'''
    train_dataset = UpperLimbMotionDataset(args.data,args.task)
    train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True) 
    
    curr_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    trainer = get_trainer_from_config(args,config,curr_date)

    '''Load checkpoint if any'''
    if args.load_ckpt is not None:
        trainer.load_weight(args.load_ckpt)

    '''Logger'''
    log = os.path.join(args.log,curr_date)
    os.makedirs(log,exist_ok=True)
    writer = SummaryWriter(log)

    '''Start Training'''
    trainer.train(train_dataloader,args.max_iter,args.save_iter,writer)



if __name__ == '__main__':
    main()