import torch
import math
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from train_utils import LinearLrDecay
from models.trainer import cGANTrainer, DiffusionTrainer, ClassifyTrainer
from models.generative.GAN import tts_cgan, eeg_cgan
from models.generative.diffusion import diffusion_ts , unet1d
from models.generative.diffusion.transformer import Transformer
from models.classification.model import InceptionTime, BasicLSTM , BasicConv1d


def get_trainer_from_config(args,config):

    if config['infra'] == 'diffusion':
        
        if config['model'] == 'unet':
            backbone = unet1d.Unet1D(**config.get('backbone',dict()))
        elif config['model'] == 'transformer':
            backbone = Transformer(**config.get('backbone',dict()))
        else:
            raise Exception("Only allow unet or transformer backbone")
        
        backbone.apply(weight_init)

        infra = diffusion_ts.Diffusion(backbone,**config.get('diffusion',dict()))
        optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, infra.parameters()),**config.get('optimizer',dict()))


        trainer = DiffusionTrainer(infra,optimizer)

    elif config['infra'] == 'gan':

        if config['model'] == 'tts':
            generator  = tts_cgan.Generator(**config.get('generator',dict()))
            discriminator = tts_cgan.Discriminator(**config.get('discriminator',dict()))
        elif config['model'] == 'eeg':
            generator = eeg_cgan.Generator(**config.get('generator',dict()))
            discriminator = eeg_cgan.Discriminator(**config.get('discriminator',dict()))
        else:
            raise Exception("Only allow tts or eeg GAN model")\
            
        generator.apply(weight_init)
        discriminator.apply(weight_init)
        g_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, generator.parameters()),**config.get('g_optim',dict()))
        d_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, discriminator.parameters()),**config.get('d_optim',dict()))
        g_scheduler = LinearLrDecay(g_optimizer,config['g_optim']['lr'],0.0,0,args.max_iter)
        d_scheduler = LinearLrDecay(d_optimizer,config['d_optim']['lr'],0.0,0,args.max_iter)
        criterion = torch.nn.CrossEntropyLoss()

        trainer = cGANTrainer(generator,discriminator,
                            g_optimizer,d_optimizer,
                            g_scheduler,d_scheduler,
                            criterion,
                            config['lambda_cls'],config['lambda_gp'],
                            args.max_iter,args.save_iter,args.n_critic,
                            config['generator']['num_classes'],config['generator']['latent_dim'])
    

    elif config['infra'] == 'classification':

        if config['model'] == 'inception-time':
            model = InceptionTime(**config.get('inception-time',dict()))

        
        if config['model'] == 'basicLSTM':
            model = BasicLSTM(**config.get('BasicLSTM',dict()))

        if config['model'] == 'basicConv1d':
            model = BasicConv1d(**config.get('BasicConv1d',dict()))

        optimizer = torch.optim.Adam(model.parameters(),**config.get('optimizer',dict()))
        criterion = torch.nn.CrossEntropyLoss()
        trainer = ClassifyTrainer(model,optimizer,criterion,config['num_classes'])


    else:
        raise Exception("Not supported type")
    

    return trainer



def weight_init(model,init_type='normal'):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1:
        if init_type == 'normal':
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif init_type == 'orth':
            nn.init.orthogonal_(model.weight.data)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform(model.weight.data, 1.)
        else:
            raise NotImplementedError('{} unknown inital type'.format(init_type))
        
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)
