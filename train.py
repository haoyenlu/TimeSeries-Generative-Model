import argparse
import PIL.Image
import torch
from tqdm import tqdm
import numpy as np
import PIL
import os

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

from utils import load_numpy_data, load_config
from dataset import UpperLimbMotionDataset
from models.tts_cgan import Generator , Discriminator
from model_utils import weight_init
from train_utils import LinearLrDecay, gradient_panelty, generate_sample_plot

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--config',type=str)
parser.add_argument('--max_iter',type=int,default=1000)
parser.add_argument('--n_critic',type=int,default=5)
parser.add_argument('--scheduler',action="store_true")
parser.add_argument('--log',type=str,default="./log")
parser.add_argument('--ckpt',type=str,default='./checkpoint')

args = parser.parse_args()
config = load_config(args.config)

(train_data,train_label) , (test_data,test_label) = load_numpy_data(args.data)

train_dataset = UpperLimbMotionDataset(train_data.transpose(0,2,1),train_label)
train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True)


gen_net  = Generator(**config.get('generator',dict()))
dis_net = Discriminator(**config.get('discriminator',dict()))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

gen_net.apply(weight_init)
dis_net.apply(weight_init)
gen_net.to(device)
dis_net.to(device)


gen_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, gen_net.parameters()),**config.get('g_optim',dict()))
dis_optimizer = torch.optim.Adam(filter(lambda p :p.requires_grad, dis_net.parameters()),**config.get('d_optim',dict()))


gen_scheduler = LinearLrDecay(gen_optimizer, config['g_optim']['lr'], 0.0, 0, args.max_iter * args.n_critic)
dis_scheduler = LinearLrDecay(dis_optimizer, config['d_optim']['lr'], 0.0, 0, args.max_iter * args.n_critic)


cls_criterion = nn.CrossEntropyLoss()

max_epoch = int(np.ceil((args.max_iter * args.n_critic) / len(train_dataloader)))

critic_step = 0
global_step = 0
writer = SummaryWriter(args.log)

'''Train Model: TODO''' 
for epoch in tqdm(range(max_epoch)):

    gen_net.train()
    dis_net.train()


    for idx, (sequence,label) in enumerate(tqdm(train_dataloader)):
        sequence = sequence.unsqueeze(2).to(device)
        label = label.to(device)

        print(sequence.size()) # Debug

        # Sample noise
        noise = torch.FloatTensor(np.random.normal(0,1,(sequence.shape[0],config['generator']['latent_dim']))).to(device)
        fake_label = torch.randint(0,config['generator']['num_classes'],(sequence.shape[0],)).to(device)

        # Generate fake sample based on fake label
        fake_sequence = gen_net(noise,fake_label)

        '''Train Discriminator'''
        critic_step += 1
        dis_net.zero_grad()
        real_out_adv , real_out_cls = dis_net(sequence)



        assert fake_sequence.size() == sequence.size(),f"fake_imgs.size(): {fake_sequence.size()} real_imgs.size(): {sequence.size()}"

        fake_out_adv , real_out_adv = dis_net(fake_sequence)
        
        # Compute loss
        alpha = torch.rand(sequence.size(0),1,1,1).to(device)
        x_hat = (alpha * sequence.data + (1-alpha) * fake_sequence.data).requires_grad_(True)
        out_src, _ = dis_net(x_hat)
        d_loss_gp = gradient_panelty(out_src,x_hat,device=device)

        d_real_loss = -torch.mean(real_out_adv)
        d_fake_loss = torch.mean(fake_out_adv)
        d_adv_loss = d_real_loss + d_fake_loss
        d_cls_loss = cls_criterion(real_out_cls,label)
        d_loss = d_adv_loss + config['lambda_cls']*d_cls_loss + config['lambda_gp'] * d_loss_gp
        d_loss.backward()

        nn.utils.clip_grad_norm_(dis_net.parameters(),5.)
        dis_optimizer.step()

        writer.add_scalar('d_loss',d_loss.item(),global_step)

        if critic_step == args.n_critic:
            '''Train Generator'''
            gen_net.zero_grad()
            fake_sequence = gen_net(noise,fake_label)
            g_out_adv, g_out_cls = dis_net(fake_sequence)

            g_adv_loss = -torch.mean(real_out_adv)
            g_cls_loss = cls_criterion(g_out_cls, fake_label)
            g_loss = g_adv_loss + config['lambda_cls'] * g_cls_loss
            g_loss.backward()

            nn.utils.clip_grad_norm_(gen_net.parameters(),5.)
            gen_optimizer.step()
            critic_step = 0

            writer.add_scalar('g_loss',g_loss.item(),global_step)

        if args.scheduler:
            g_lr = gen_scheduler.step(global_step) # Should generator lr scheduler decay every step ? 
            d_lr = dis_scheduler.step(global_step)
            writer.add_scalar('LR/g_lr',g_lr,global_step)
            writer.add_scalar('LR/d_lr',d_lr,global_step)
        
        '''Moving Average Weight (Not understand yet): TODO'''

        global_step += 1
        tqdm.set_description(f"[Epoch:{epoch}/{max_epoch}][Batch:{idx}/{len(train_dataloader)}][d_loss:{d_loss.item()}][g_loss:{g_loss.item()}]")


    '''Visualize Synthesize Sample'''
    gen_net.eval()
    plot_buf = generate_sample_plot(gen_net,config,epoch=epoch)
    image = PIL.Image.open(plot_buf)
    image = ToTensor()(image).unsqueeze(0)
    writer.add_image('Image',image[0],epoch)


    '''Save Checkpoint'''
    ckpt = {
        'epoch':epoch+1,
        'gen_state_dict':gen_net.state_dict(),
        'dis_state_dict':dis_net.state_dict(),
        'gen_optim': gen_optimizer.state_dict(),
        'dis_optim': dis_optimizer.state_dict(),
    }

    torch.save(ckpt,os.path.join(args.ckpt,'checkpoint.pth'))

