import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from utils import load_numpy_data, load_config
from dataset import UpperLimbMotionDataset
from model_utils import weight_init
from train_utils import LinearLrDecay
from models.trainer import cGANTrainer, DiffusionTrainer
from models.generative.GAN import tts_cgan, eeg_cgan
from models.generative.diffusion import diffusion_ts , unet1d
from models.generative.diffusion.transformer import Transformer

from argument import train_argument



def get_trainer_GAN(args,config,curr_date):
    # generator  = tts_cgan.Generator(**config.get('generator',dict()))
    # discriminator = tts_cgan.Discriminator(**config.get('discriminator',dict()))
    '''EEG GAN'''
    generator = eeg_cgan.Generator(**config.get('generator',dict()))
    discriminator = eeg_cgan.Discriminator(**config.get('discriminator',dict()))
    generator.apply(weight_init)
    discriminator.apply(weight_init)
    g_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, generator.parameters()),**config.get('g_optim',dict()))
    d_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, discriminator.parameters()),**config.get('d_optim',dict()))
    g_scheduler = LinearLrDecay(g_optimizer,config['g_optim']['lr'],0.0,0,args.max_iter)
    d_scheduler = LinearLrDecay(d_optimizer,config['d_optim']['lr'],0.0,0,args.max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    '''Logger'''
    log = os.path.join(args.log,curr_date)
    os.makedirs(log,exist_ok=True)
    writer = SummaryWriter(log)

    '''Checkpoint'''    
    ckpt = os.path.join(args.ckpt,curr_date)
    os.makedirs(ckpt,exist_ok=True)

    trainer = cGANTrainer(generator,discriminator,
                      g_optimizer,d_optimizer,
                      g_scheduler,d_scheduler,
                      criterion,
                      config['lambda_cls'],config['lambda_gp'],
                      args.max_iter,args.save_iter,args.n_critic,
                      config['generator']['num_classes'],config['generator']['latent_dim'],
                      writer,ckpt)
    return trainer

def get_trainer_diffusion(args,config,curr_date):
    
    if args.backbone == 'unet':
        backbone = unet1d.Unet1D(**config.get('backbone',dict()))
    elif args.backbone == 'transformer':
        backbone = Transformer(**config.get('backbone',dict()))
    else:
        raise Exception("Only allow unet or transformer backbone")
    

    diffusion_model = diffusion_ts.Diffusion(backbone,**config.get('diffusion',dict()))
    backbone.apply(weight_init)

    optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, diffusion_model.parameters()),**config.get('optimizer',dict()))
    scheduler = LinearLrDecay(optimizer,config['optimizer']['lr'],0.0,0,args.max_iter)
    
    '''Logger'''
    log = os.path.join(args.log,curr_date)
    os.makedirs(log,exist_ok=True)
    writer = SummaryWriter(log)

    '''Checkpoint'''    
    ckpt = os.path.join(args.ckpt,curr_date)
    os.makedirs(ckpt,exist_ok=True)

    trainer = DiffusionTrainer(diffusion_model,optimizer,
                        scheduler,args.max_iter,args.save_iter,
                        ckpt,writer)
    
    return trainer



def main():
    args = train_argument()
    config = load_config(args.config)

    '''Load Data'''
    train_dataset = UpperLimbMotionDataset(args.data,args.task)
    train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True) 
    
    curr_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    # trainer = get_trainer_GAN(args,config,curr_date)
    trainer = get_trainer_diffusion(args,config,curr_date)

    '''Load checkpoint if any'''
    if args.load_ckpt is not None:
        trainer.load_weight(args.load_ckpt)


    '''Start Training'''
    trainer.train(train_dataloader)



if __name__ == '__main__':
    main()