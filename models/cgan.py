import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from train_utils import gradient_panelty

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


class ConditionalGAN:
    def __init__(self,generator,discriminator,g_optimizer,d_optimizer,
                 criterion,lambda_cls,lambda_gp,
                 max_iter,save_iter,n_critic,num_classes,latent_dim,
                 writer):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.criterion = criterion
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.lambda_gp = lambda_gp
        self.lamdba_cls = lambda_cls
        self.max_iter = max_iter
        self.save_iter = save_iter
        self.n_critic = n_critic
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.writer = writer


    def train(self,dataloader):
        '''Train Coniditional GAN model'''
        dataloader_cycle = cycle(dataloader)

        self.generator.train()
        self.discriminator.train()

        for iter in tqdm(range(self.max_iter)):
            sequence , label = next(dataloader_cycle)
            sequence = sequence.to(self.device)
            label = label.to(self.device)
            onehot_label = F.one_hot(label.squeeze().long(),num_classes=self.num_classes).float()


            '''Train Discriminator'''
            for d_step in range(self.n_critic):
                '''Noise and Fake Sequence'''
                noise = torch.FloatTensor(np.random.normal(0,1,(sequence.shape[0],self.latent_dim))).to(self.device)
                fake_label = torch.randint(0,self.num_classes,(sequence.shape[0],)).to(self.device)
                fake_sequence = self.generator(noise,fake_label)
                
                self.discriminator.zero_grad()
                real_out_adv, real_out_cls = self.discriminator(sequence)
                fake_out_adv, fake_out_cls = self.discriminator(fake_sequence)

                '''Compute Critic Loss (Temperary)'''
                # Gradient Penalty Loss
                alpha = torch.rand(sequence.size(0),1,1).to(self.device)
                x_hat = (alpha * sequence.data + (1-alpha) * fake_sequence).requires_grad_(True)
                out_src, _ = self.discriminator(x_hat)
                d_loss_gp = gradient_panelty(out_src,x_hat,device=self.device)

                d_real_loss = -torch.mean(real_out_adv)
                d_fake_loss = torch.mean(fake_out_adv)
                d_adv_loss = d_real_loss + d_fake_loss
                d_cls_loss = self.criterion(real_out_cls,onehot_label)
                d_loss = d_adv_loss + self.lambda_gp * d_loss_gp + self.lamdba_cls * d_cls_loss
                d_loss.backward()

                # Clip Weight
                nn.utils.clip_grad_norm(self.discriminator.parameters(),5.)
                self.d_optimizer.step()

                tqdm.write(f"[Critic Step:{d_step}/{self.n_critic}][d_loss:{d_loss.item()}]")
            
            '''Train Generator'''
            self.generator.zero_grad()
            noise = torch.FloatTensor(np.random.normal(0,1,(sequence.shape[0],self.latent_dim))).to(self.device)
            fake_label = torch.randint(0,self.num_classes,(sequence.shape[0],)).to(self.device)
            fake_sequence = self.generator(noise,fake_label)
            g_out_adv, g_out_cls = self.discriminator(fake_sequence)
            g_adv_loss = -torch.mean(g_out_adv)
            g_cls_loss = self.criterion(g_out_cls,fake_label)

            tqdm.write(f"[Iteration:{iter}/{self.max_iter}][g_loss:{g_los}]")


