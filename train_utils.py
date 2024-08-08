import torch

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
    

class ExponentialLrDecay(object):
    def __init__(self,optimizer, global_step, start_lr):
        pass


