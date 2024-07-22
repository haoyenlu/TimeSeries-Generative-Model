import PIL.Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import PIL
from torchvision.transforms import ToTensor
import math

from train_utils import gradient_panelty

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


class ConditionalGAN:
    def __init__(self,generator,discriminator,
                 g_optimizer,d_optimizer,
                 g_scheduler,d_scheduler,
                 criterion,lambda_cls,lambda_gp,
                 max_iter,save_iter,n_critic,num_classes,latent_dim,
                 writer,save_path):
        
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler
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
        self.save_path = save_path

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

                # Wasserstein Loss
                d_real_loss = -torch.mean(real_out_adv)
                d_fake_loss = torch.mean(fake_out_adv)
                d_adv_loss = d_real_loss + d_fake_loss
                d_cls_loss = self.criterion(real_out_cls,onehot_label)
                d_loss = d_adv_loss + self.lambda_gp * d_loss_gp + self.lamdba_cls * d_cls_loss
                d_loss.backward()

                # Clip Weight
                nn.utils.clip_grad_norm_(self.discriminator.parameters(),10.)
                self.d_optimizer.step()

                self.writer.add_scalar('d_loss',d_loss.item(),iter*self.n_critic + d_step)
                tqdm.write(f"[Critic Step:{d_step}/{self.n_critic}][d_loss:{d_loss.item()}]")
                
            '''Train Generator'''
            self.generator.zero_grad()
            noise = torch.FloatTensor(np.random.normal(0,1,(sequence.shape[0],self.latent_dim))).to(self.device)
            fake_label = torch.randint(0,self.num_classes,(sequence.shape[0],)).to(self.device)
            fake_onehot_label = F.one_hot(fake_label.squeeze().long(),num_classes=self.num_classes).float()
            fake_sequence = self.generator(noise,fake_label)
            g_out_adv, g_out_cls = self.discriminator(fake_sequence)
            g_adv_loss = -torch.mean(g_out_adv)
            g_cls_loss = self.criterion(g_out_cls,fake_onehot_label)
            g_loss = g_adv_loss + self.lamdba_cls*g_cls_loss
            g_loss.backward()

            # Clip Weight
            nn.utils.clip_grad_norm_(self.generator.parameters(),10.)
            self.g_optimizer.step()

            self.writer.add_scalar('g_loss',g_loss.item(),iter)
            tqdm.write(f"[Iteration:{iter}/{self.max_iter}][g_loss:{g_loss.item()}]")
            
            
            '''LR scheduler step'''
            g_lr = self.g_scheduler.step(iter)
            d_lr = self.d_scheduler.step(iter)
            self.writer.add_scalar('LR/g_lr',g_lr,iter)
            self.writer.add_scalar('LR/d_lr',d_lr,iter)    


            if (iter+1) % self.save_iter == 0:
                '''Visualize SYnthetic data'''
                plot_buf = self.visualize(iter)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).unsqueeze(0)
                self.writer.add_image('Image',image[0],iter)

                self.save_weight(iter)



    def save_weight(self,epoch):
        ckpt = {
            'epoch':epoch+1,
            'gen_state_dict':self.generator.state_dict(),
            'dis_state_dict':self.discriminator.state_dict(),
            'gen_optim':self.g_optimizer.state_dict(),
            'dis_optim':self.d_optimizer.state_dict()
        }

        torch.save(ckpt,os.path.join(self.save_path,'checkpoint.pth'))


    
    def visualize(self,epoch):
        self.generator.eval()
        num_sample = 6
        noise = torch.FloatTensor(np.random.normal(0,1,(num_sample,self.latent_dim))).to(self.device)
        fake_label = torch.randint(0,self.num_classes,(num_sample,))
        fake_sequence = self.generator(noise,fake_label.to(self.device)).to('cpu').detach().numpy()
        _,c,_ = fake_sequence.shape

        fig, axs = plt.subplots(2,3,figsize=(20,8))
        fig.suptitle(f'Synthetic data at epoch {epoch}',fontsize=20)

        for i in range(2):
            for j in range(3):
                for k in range(c):
                    axs[i, j].plot(fake_sequence[i*3+j][k][:])
            
                axs[i, j].title.set_text(fake_label[i*3+j].item())

        buf = io.BytesIO()
        plt.savefig(buf,format='jpg')
        plt.close(fig)
        buf.seek(0)
        return buf


    def load_weight(self,checkpoint):
        ckpt = torch.load(checkpoint,map_location=self.device)
        self.generator.load_state_dict(ckpt['gen_state_dict'])
        self.discriminator.load_state_dict(ckpt['dis_state_dict'])
        self.g_optimizer.load_state_dict(ckpt['gen_optim'])
        self.d_optimizer.load_state_dict(ckpt['dis_optim'])


    def generate_sample(self,num_samples = 1000, sample_per_batch = 10):
        samples = []
        iter = math.floor(num_samples / sample_per_batch)
        res = num_samples % sample_per_batch

        for i in range(iter):
            noise = torch.FloatTensor(np.random.normal(0,1,(sample_per_batch,self.latent_dim))).to(self.device)
            fake_label = torch.randint(0,self.num_classes,(sample_per_batch,)).to(self.device)
            fake_sequence = self.generator(noise,fake_label).to('cpu').detach().numpy()
            samples.append(fake_sequence)
        
        if res:
            noise = torch.FloatTensor(np.random.normal(0,1,(res,self.latent_dim))).to(self.device)
            fake_label = torch.randint(0,self.num_classes,(res,)).to(self.device)
            fake_sequence = self.generator(noise,fake_label).to('cpu').detach().numpy()
            samples.append(fake_sequence)

        return np.array(samples).squeeze()
    