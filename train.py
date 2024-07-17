import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import load_numpy_data, load_config
from dataset import UpperLimbMotionDataset
from models import tts_cgan
from model_utils import weight_init
from train_utils import LinearLrDecay, gradient_panelty, generate_sample_plot
from models.cgan import ConditionalGAN


parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--config',type=str)
parser.add_argument('--max_iter',type=int,default=1000)
parser.add_argument('--save_iter',type=int,default=100)
parser.add_argument('--n_critic',type=int,default=5)
parser.add_argument('--scheduler',action="store_true")
parser.add_argument('--log',type=str,default="./log")
parser.add_argument('--ckpt',type=str,default='./checkpoint')

args = parser.parse_args()
config = load_config(args.config)

(train_data,train_label) , (test_data,test_label) = load_numpy_data(args.data)
train_dataset = UpperLimbMotionDataset(train_data.transpose(0,2,1),train_label)
train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True)


generator  = tts_cgan.Generator(**config.get('generator',dict()))
discriminator = tts_cgan.Discriminator(**config.get('discriminator',dict()))
generator.apply(weight_init)
discriminator.apply(weight_init)

g_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, generator.parameters()),**config.get('g_optim',dict()))
d_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, discriminator.parameters()),**config.get('d_optim',dict()))
g_scheduler = LinearLrDecay(g_optimizer,config['g_optim']['lr'],0.0,0,args.max_iter)
d_scheduler = LinearLrDecay(d_optimizer,config['d_optim']['lr'],0.0,0,args.max_iter)
criterion = torch.nn.CrossEntropyLoss()
writer = SummaryWriter(args.log)


cgan = ConditionalGAN(generator,discriminator,
                      g_optimizer,d_optimizer,
                      g_scheduler,d_scheduler,
                      criterion,
                      config['lambda_cls'],config['lambda_gp'],
                      args.max_iter,args.save_iter,args.n_critic,
                      config['generator']['num_classes'],config['generator']['latent_dim'],
                      writer,args.ckpt)

cgan.train(train_dataloader)
