import torch
import torch.nn as nn


from models.generative.diffusion.transformer_utils import AdaLayerNorm


class DownSampleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,emb_dim,kernel=3,padding="same",downsample=True,norm=True):
        super(DownSampleBlock,self).__init__()
        self.norm = norm
        self.ln = AdaLayerNorm(emb_dim)

        modules = []

        modules.append(nn.Conv1d(in_ch,out_ch,kernel=kernel,padding=padding))
        modules.append(nn.LeakyReLU(0.2))
        if downsample:
            modules.append(nn.AvgPool1d(kernel_size=3,padding=1,stride=2))

        self.modules =  nn.Sequential(*modules)
    
    def forward(self,x,timestep,emb_label):
        if self.norm:
            x = self.ln(x,timestep,emb_label)

        x = self.modules(x)
        return x
        

class UpsampleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,emb_dim,kernel=3,padding="same",upsample=True,norm=True):
        super(UpsampleBlock,self).__init__()
        self.norm = norm
        self.ln = AdaLayerNorm(emb_dim)

        modules = []
        if upsample:
            modules.append(nn.Upsample(scale_factor=2))
        
        modules.append(nn.Conv1d(in_ch,out_ch,kernel,padding="same"))
        # block.append(nn.BatchNorm1d(out_ch))
        modules.append(nn.LeakyReLU(0.2))

        self.modules =  nn.Sequential(*modules)

    def forward(self,x,timestep,emb_label):
        if self.norm:
            x = self.ln(x,timestep,emb_label)

        x = self.modules(x)
        return x


class Unet1D(nn.Module):
    def __init__(self,seq_len,feature_dim,num_classes,hidden_ch,emb_dim):
        super(Unet1D,self).__init__()

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_ch
        self.emb_dim = emb_dim

        self.label_emb = nn.Embedding(num_classes,emb_dim)

        modules = []
        
        '''Encoder (Downsample)'''
        prev_ch = feature_dim
        for i in range(6):
            modules.append(
                DownSampleBlock(in_ch=prev_ch,out_ch=hidden_ch,emb_dim=emb_dim,kernel=3,padding="same",downsample=False,norm=True),
            )
            modules.append(
                DownSampleBlock(in_ch=hidden_ch,out_ch=hidden_ch,emb_dim=emb_dim,kernel=3,padding="same",downsample=True,norm=False)
            )
            prev_ch = hidden_ch
        
        '''Decoder (Upsample)'''
        for i in range(6):
            modules.append(
                UpsampleBlock(in_ch=hidden_ch * 2,out_ch=hidden_ch,emb_dim=emb_dim,kernel=3,padding="same",upsample=False,norm=True),
            )
            modules.append(
                UpsampleBlock(in_ch=hidden_ch,out_ch=hidden_ch,emb_dim=emb_dim,kernel=3,padding="same",upsample=True,norm=False),
            )

        self.modules = nn.ModuleList(modules)

        self.last = nn.Sequential(
            nn.Conv1d(hidden_ch,feature_dim,kernel_size=5,padding="same"),
            nn.Sigmoid()
        )

    

    def forward(self,x,timestep,label): # Pytorch (N,C,L)
        label_embedding  = self.label_emb(label)

        _x = x
        res = []
        # Encoding
        cnt = 0
        while cnt < 6:
            _x = self.modules[cnt*2](_x,timestep,label_embedding)
            res.append(_x)
            _x = self.modules[cnt*2 + 1](_x,timestep,label_embedding)
            cnt += 1
        
        # Decoding
        res_cnt = 5
        while cnt < 12:
            _x = torch.concat([_x,res[res_cnt]],dim=1)
            _x = self.modules[cnt*2](_x,timestep,label_embedding)
            _x = self.modules[cnt*2 + 1](_x,timestep,label_embedding)
            res_cnt -= 1
            cnt += 1
        
        _x = self.last(_x)
        return _x


        

            



    

