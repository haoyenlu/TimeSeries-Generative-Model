import torch
import math
import torch.nn as nn
import numpy as np

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


def generate_samples(model,num_samples=1000,sample_per_batch=10):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        samples = []
        labels = []

        latent_dim = model.latent_dim
        num_classes = model.num_classes

        model.to(device)
        model.eval()

        cnt = 0
        while cnt < num_samples:
            num_per_batch = min(num_per_batch,num_samples - cnt)
            noise = torch.FloatTensor(np.random.normal(0,1,(num_per_batch,latent_dim))).to(device)
            fake_label = torch.randint(0,num_classes,(num_per_batch,)).to(device)
            fake_sequence = model(noise,fake_label).to('cpu').detach().numpy()
            samples.append(fake_sequence)
            labels.append(fake_label.to('cpu').detach().numpy())
            cnt += num_per_batch

        return np.array(samples).squeeze(), np.array(labels).squeeze()