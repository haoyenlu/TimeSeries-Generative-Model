import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np

from utils import load_config
from dataset import UpperLimbMotionDataset
from model_utils import  get_trainer_from_config
from train_utils import LinearLrDecay

from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample

from argument import train_argument







def main():
    args = train_argument()
    config = load_config(args.config)

    if args.curr_date is None:
        curr_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    else:
        curr_date = args.curr_date


    for task in args.task:
        '''Load Data'''
        train_dataset = UpperLimbMotionDataset(args.data,task)
        train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True) 
        

        trainer = get_trainer_from_config(args,config,curr_date)

        # load checkpoint
        if args.load_ckpt is not None:
            trainer.load_weight(args.load_ckpt)

        # logger
        log = os.path.join(args.log,curr_date,task)
        os.makedirs(log,exist_ok=True)
        writer = SummaryWriter(log)
        
        # Learning rate scheduler
        scheduler = LinearLrDecay(trainer.optimizer,config['optimizer']['lr'],0.0,0,args.max_iter)


        # training
        trainer.train(train_dataloader,args.max_iter,args.save_iter,scheduler,writer)


        # generate samples
        samples = trainer.generate_samples(num_samples=100,num_per_batch=10)
        save_path = os.path.join(args.save,curr_date,task)
        os.makedirs(save_path,exist_ok=True)
        np.save(os.path.join(save_path,'synthesize.npy'), samples)

        # analyze with pca and tsne
        train_data = train_dataset._get_numpy()

        plot_pca(real=train_data,fake=samples,save_path=save_path)
        plot_tsne(real=train_data,fake=samples,save_path=save_path)
        plot_umap(real=train_data,fake=samples,save_path=save_path)
        plot_sample(real=train_data,fake=samples,save_path=save_path)

if __name__ == '__main__':
    main()