import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from utils import load_numpy_data, load_config
from dataset import UpperLimbMotionDataset
from train_utils import LinearLrDecay
from models.generative.trainer import DiffusionTrainer
from models.generative.diffusion import diffusion_ts, unet1d


parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--config',type=str)
parser.add_argument('--max_iter',type=int,default=1000)
parser.add_argument('--save_iter',type=int,default=100)
parser.add_argument('--log',type=str,default="./log")
parser.add_argument('--ckpt',type=str,default='./checkpoint')
parser.add_argument('--load_ckpt',type=str,default=None)

args = parser.parse_args()
config = load_config(args.config)

(train_data,train_label) , (test_data,test_label) = load_numpy_data(args.data)
train_dataset = UpperLimbMotionDataset(train_data.transpose(0,2,1),train_label)
train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True)

backbone = unet1d.Unet1D(**config.get('backbone',dict()))
diffusion_model = diffusion_ts.Diffusion(backbone,**config.get('diffusion',dict()))


optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, diffusion_model.parameters()),**config.get('optimizer',dict()))
scheduler = LinearLrDecay(optimizer,config['optimizer']['lr'],0.0,0,args.max_iter)

cur_date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
log = os.path.join(args.log,cur_date)
os.makedirs(log,exist_ok=True)

writer = SummaryWriter(log)

ckpt = os.path.join(args.ckpt,cur_date)
os.makedirs(ckpt,exist_ok=True)


# Conditional GAN architecture
trainer = DiffusionTrainer(diffusion_model,optimizer,
                        scheduler,args.max_iter,args.save_iter,
                        ckpt,writer)


# Load Checkpoint if any
if args.load_ckpt is not None:
    trainer.load_weight(args.load_ckpt)


# Train model
trainer.train(train_dataloader)
