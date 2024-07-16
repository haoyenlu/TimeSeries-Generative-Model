import torch
import numpy as np
import matplotlib.pyplot as plt
import io

'''L2 Norm'''
def gradient_panelty(y,x,device):
    weights = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                          inputs=x,
                          grad_outputs=weights,
                          retain_graph=True,
                          create_graph=True,
                          only_inputs=True)[0]
    
    dydx = dydx.reshape(dydx.size(0),-1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2,dim=1))
    return torch.mean((dydx_l2norm-1)**2)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
    


def generate_sample_plot(gen_net,config,epoch,num=6):
    '''Generate pyplot and save to buffer'''
    assert num % 2 == 0, "Number of sample has to be divisible by 2"

    synthetic_data = []
    synthetic_label = []

    for i in range(num):
        noise = torch.FloatTensor(np.random.normal(0,1,(1,config['generator']['latent_dim'])))
        fake_label = torch.randint(0,config['g_optim']['num_classes'],(1,))
        fake_sequence = gen_net(noise,fake_label).to('cpu').detach().numpy()

        synthetic_data.append(fake_sequence)
        synthetic_label.append(fake_label)
    
    fig, axs = plt.subplots(2,num//2,figsize=(20,8))
    fig.suptitle(f'Synthetic data at epoch {epoch}',fontsize=20)

    for i in range(2):
        for j in range(num//2):
            axs[i, j].plot(synthetic_data[i*(num//2)+j][0][0][0][:])
            axs[i, j].title.set_text(synthetic_label[i*(num//2)+j])

    buf = io.BytesIO()
    plt.savefig(buf,format='jpg')
    buf.seek(0)
    return buf
