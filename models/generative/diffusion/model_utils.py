import math
import torch
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, reduce
from scipy.fftpack import next_fast_len


def exists(x):
    return x is not None

def default(val,d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t,*args,**kwargs):
    return t

def extract(a,t,x_shape):
    b, *_ = t.shape
    out = a.gather(-1,t)
    return out.reshape(b,*((1,) * (len(x_shape) - 1)))

def Upsample(dim,dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode="nearest"),
        nn.Conv1d(dim,default(dim_out,dim),3,padding=1)
    )

def Downsample(dim,dim_out=None):
    return nn.Conv1d(dim,default(dim_out,dim),4,2,1)

# normalization function

def normalize_to_neg_one_to_one(x):
    return x*2 -1

def unnormalize_to_zero_to_one(x):
    return (x+1) * 0.5

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
    
    def forward(self,x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim,device=device) * -emb)
        emb = x[:,None] * emb[None,:]
        emb = torch.cat((emb.sin(),emb.cos()),dim = -1)
        return emb
    
# learnable positional emb

class LearnablePositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=1024):
        super(LearnablePositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1,max_len,d_model))
        nn.init.uniform_(self.pe,-0.02,0.02)

    def forward(self,x):
        x = x + self.pe
        return self.dropout(x)
    
class moving_avg(nn.Module):
    """
    Moving Average block to highlight the trend of time series
    """
    def __init__(self,kernel_size,stride):
        super(moving_avg,self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,stride=stride,padding=0)

    def forward(self,x):
        front = x[:,0:1,:].repeat(1,self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2),1)
        end = x[:,-1:,:].repeat(1,math.floor((self.kernel_size - 1) // 2),1)
        x = torch.cat([front, x , end], dim = 1)
        x = self.avg(x.permute(0,2,1))
        x = x.permute(0,2,1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self,kernel_size):
        super(series_decomp,self).__init__()
        self.moving_avg = moving_avg(kernel_size,stride=1)
    
    def forward(self,x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self,kernel_size):
        super(series_decomp_multi,self).__init__()
        self.moving_avg = [moving_avg(kernel,stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1,len(kernel_size))

    def forward(self,x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean
    
class Transpose(nn.Module):
    def __init__(self,shape:tuple):
        super(Transpose,self).__init__()
        self.shape = shape
    
    def forward(self,x):
        return x.transpose(*self.shape)
    
class Conv_MLP(nn.Module):
    def __init__(self,in_dim,out_dim,resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1,2)),
            nn.Conv1d(in_dim,out_dim,3,stride=1,padding=1),
            nn.Dropout(p=resid_pdrop)
        )    
    
    def forward(self,x):
        return self.sequential(x).transpose(1,2)

class Transformer_MLP(nn.Module):
    def __init__(self,n_embd,mlp_hidden_times,act,resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=n_embd,out_channels=int(mlp_hidden_times * n_embd),kernel_size=1,padding=0),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd),out_channels=int(mlp_hidden_times * n_embd),kernel_size=3,padding=1),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd),out_channels=n_embd,kernel_size=3,padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self,x):
        return self.sequential(x)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x * F.sigmoid(1.702 * x)
    

class AdaLayerNorm(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd,n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd,elementwise_affine=False)
    
    def forward(self,x,timestep,label_emb=None):
        emb = self.emb(timestep) + label_emb if label_emb is not None else self.emb(timestep)
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb,2,dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(nn.Module):
    def __init__(self,in_ch,num_classes,eps=1e-7):
        super().__init__()
        self.num_classes =num_classes
        self.in_ch = in_ch
        self.eps = eps
        self.label_emb = nn.Embedding(num_classes,in_ch*4)
        self.timestep_emb = SinusoidalPosEmb(in_ch*4)
    
    def c_norm(self,x,bs,ch,eps=1e-7):
        assert isinstance(x,torch.cuda.FloatTensor)
        x_var = x.var(dim=-1)
        x_std = x_var.sqrt().view(bs,ch,1)
        x_mean = x.mean(dim=-1).view(bs,ch,1)
        return x_std, x_mean

    def forward(self,x,timestep,label=None):
        size = x.size()
        bs ,ch = size[:2]
        x_ = x.view(bs,ch,-1)
        emb = self.timestep_emb(timestep)
        emb = emb + self.label_emb(label) if label is not None else emb
        emb = emb.view(bs,ch,-1)
        x_std, x_mean = self.c_norm(x_,bs,ch,eps=self.eps)
        y_std,y_mean = self.c_norm(emb,bs,ch,eps=self.eps)

        out = ((x-x_mean.expand(size))/ x_std.expand(size)) * y_std.expand(size) + y_mean.expand(size)
        return out

    