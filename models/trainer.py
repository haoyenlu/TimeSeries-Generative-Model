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
from abc import ABC , abstractmethod


from train_utils import gradient_panelty

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def sample2buffer(samples,labels,iter,num_samples=6):
    _,c,_ = samples.shape
    fig, axs = plt.subplots(2,3,figsize=(20,8))
    fig.suptitle(f'Synthetic data at epoch {iter}',fontsize=20)

    for i in range(2):
        for j in range(3):
            for k in range(c):
                axs[i, j].plot(samples[i*3+j][k][:])
        
            axs[i, j].title.set_text(labels[i*3+j])

    buf = io.BytesIO()
    plt.savefig(buf,format='jpg')
    plt.close(fig)
    buf.seek(0)
    return buf



class BaseTrainer(ABC):

    @abstractmethod
    def train(self,dataloader):
        pass

    @abstractmethod
    def save_weight(self,iter):
        pass

    @abstractmethod
    def load_weight(self,ckpt):
        pass


class cGANTrainer(BaseTrainer):
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


    def load_weight(self,ckpt):
        if ckpt is None: return 
        data = torch.load(os.path.join(ckpt,"checkpoint.pth"),map_location=self.device)
        self.generator.load_state_dict(data['gen_state_dict'])
        self.discriminator.load_state_dict(data['dis_state_dict'])
        self.g_optimizer.load_state_dict(data['gen_optim'])
        self.d_optimizer.load_state_dict(data['dis_optim'])


    def visualize(self,epoch):
        self.generator.eval()
        num_sample = 6
        noise = torch.FloatTensor(np.random.normal(0,1,(num_sample,self.latent_dim))).to(self.device)
        fake_label = torch.randint(0,self.num_classes,(num_sample,))
        fake_sequence = self.generator(noise,fake_label.to(self.device)).to('cpu').detach().numpy()
        buf = sample2buffer(fake_sequence,fake_label.item(),epoch)
        return buf
    



class DiffusionTrainer(BaseTrainer):
    def __init__(self,model,optimizer,scheduler,
                 max_iter,save_iter,
                 save_path,writer):
        
        self.model = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.max_iter = max_iter
        self.save_iter = save_iter
        self.save_path = save_path

        self.model.to(self.device)

    def train(self,dataloader):
        dataloader_cycle = cycle(dataloader)

        self.model.train()

        for iter in tqdm(range(self.max_iter)):
            self.model.zero_grad()
            sequence = next(dataloader_cycle)
            sequence = sequence.to(self.device)
            # label = label.to(self.device)

            loss = self.model(sequence,target=sequence,label=None)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),5.)
            self.optimizer.step()
            lr = self.scheduler.step(iter)

            self.writer.add_scalar('loss',loss.item(),iter)
            self.writer.add_scalar('lr',lr,iter)
            tqdm.write(f"[Iter:{iter}/{self.max_iter}][loss:{loss.item()}]")

            if (iter+1) % self.save_iter == 0:
                '''Visualize SYnthetic data'''
                plot_buf = self.visualize(iter)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).unsqueeze(0)
                self.writer.add_image('Image',image[0],iter)
                self.save_weight(iter)
            

    def save_weight(self, iter):
        data = {
            'epoch': iter + 1,
            'model':self.model.state_dict(),
            'opt': self.optimizer.state_dict()
        }
        torch.save(data,os.path.join(self.save_path,"checkpoint.pth"))


    def load_weight(self, ckpt):
        if ckpt is None: return
        data = torch.load(ckpt)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])


    def visualize(self,iter):
        self.model.eval()
        num_samples = 6
        samples , labels = self.model.generate_mts(batch_size=num_samples,use_label=True)
        samples = samples.to('cpu').detach().numpy()
        labels = labels.to('cpu').detach().numpy()
        buf = sample2buffer(samples,labels,iter)
        return buf
        
    


class ClassifyTrainer(BaseTrainer):
    def __init__(self,model,optimizer,criterion,
                 scheduler,
                 num_classes,
                 max_iter,
                 writer,save_path):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_iter = max_iter
        self.writer = writer
        self.save_path = save_path
        self.num_classes = num_classes

        self.model.to(self.device)

    
    def train(self, train_dataloader, test_dataloader):
        n = len(train_dataloader)
        m = len(test_dataloader)

        for iter in tqdm(range(self.max_iter)):
            train_total_loss = 0
            test_total_loss = 0
            train_total_accuracy = 0
            test_total_accuracy = 0

            '''Train model with train dataset'''
            self.model.train()
            for train_data, train_label in train_dataloader:
                self.model.zero_grad()
                train_data = train_data.to(self.device)
                train_label = train_label.to(self.device)
                onehot_label = F.one_hot(train_label.long(),num_classes=self.num_classes).float()
                pred = self.model(train_data)
                train_loss = self.criterion(pred,onehot_label)
                train_loss.backward()
                train_total_loss += train_loss.item()
                train_total_accuracy += (torch.argmax(pred) == train_label).sum().item()
                self.optimizer.step()

            '''Evaluate model with test dataset'''
            self.model.eval()
            for test_data, test_label in test_dataloader:
                test_data = test_data.to(self.device)
                test_label = test_label.to(self.device)
                onehot_label = F.one_hot(train_label.long(),num_classes=self.num_classes).float()
                pred = self.model(train_data)
                test_loss = self.criterion(pred,onehot_label)
                test_total_loss += test_loss.item()
                test_total_accuracy += (torch.argmax(pred) == test_label).sum().item()

            
            
            # lr = self.scheduler.step(iter)
            tqdm.write(f"[Epoch:{iter}/{self.max_iter}][Train Loss:{train_total_loss/n:.4f}][Train Accuracy:{train_total_accuracy/n:.4f}][Test Loss:{test_total_loss/m:.4f}][Test Accuracy:{test_total_accuracy/m:.4f}]")

            self.writer.add_scalar('loss/train_loss',train_total_loss/n,iter)
            self.writer.add_scalar('loss/test_loss',test_total_loss/m,iter)
            self.writer.add_scalar('accuracy/train_accuracy',train_total_accuracy/n,iter)
            self.writer.add_scalar('accuracy/test_accuracy',test_total_accuracy/m,iter)
            # self.writer.add_scalar('lr',lr,iter)


            self.save_weight(iter)
    

    def save_weight(self, iter):
        data = {
            'iter':iter,
            'model':self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        
        torch.save(data,os.path.join(self.save_path,"checkpoint.pth"))
    
    def load_weight(self, ckpt):
        data = torch.load(ckpt,map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
    
    


