import torch.nn as nn


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