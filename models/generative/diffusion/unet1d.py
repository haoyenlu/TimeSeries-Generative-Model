import torch
import torch.nn as nn


from models.generative.diffusion.transformer_utils import AdaInsNorm


class DownSampleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,num_classes,kernel=3,padding="same",downsample=True,norm=True):
        super(DownSampleBlock,self).__init__()
        self.norm = norm
        self.ln = AdaInsNorm(in_ch,num_classes)

        blocks = []

        blocks.append(nn.Conv1d(in_ch,out_ch,kernel_size=kernel,padding=padding))
        blocks.append(nn.LeakyReLU(0.2))
        if downsample:
            blocks.append(nn.AvgPool1d(kernel_size=3,padding=1,stride=2))

        self.blocks =  nn.Sequential(*blocks)
    
    def forward(self,x,timestep,emb_label):
        if self.norm:
            x = self.ln(x,timestep,emb_label)

        x = self.blocks(x)
        return x
        

class UpsampleBlock(nn.Module):
    def __init__(self,in_ch,out_ch,num_classes,kernel=3,padding="same",upsample=True,norm=True):
        super(UpsampleBlock,self).__init__()
        self.norm = norm
        self.ln = AdaInsNorm(in_ch,num_classes)

        blocks = []
        if upsample:
            blocks.append(nn.Upsample(scale_factor=2))
        
        blocks.append(nn.Conv1d(in_ch,out_ch,kernel_size=kernel,padding="same"))
        # block.append(nn.BatchNorm1d(out_ch))
        blocks.append(nn.LeakyReLU(0.2))

        self.blocks =  nn.Sequential(*blocks)

    def forward(self,x,timestep,emb_label):
        if self.norm:
            x = self.ln(x,timestep,emb_label)

        x = self.blocks(x)
        return x


class Unet1D(nn.Module):
    def __init__(self,seq_len,feature_dim,num_classes,hidden_ch,emb_dim):
        super(Unet1D,self).__init__()

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_ch
        self.emb_dim = emb_dim

        blocks = []
        
        '''Encoder (Downsample)'''
        prev_ch = feature_dim
        for i in range(6):
            blocks.append(
                DownSampleBlock(in_ch=prev_ch,out_ch=hidden_ch,num_classes=num_classes,kernel=3,padding="same",downsample=False,norm=True),
            )
            blocks.append(
                DownSampleBlock(in_ch=hidden_ch,out_ch=hidden_ch,num_classes=num_classes,kernel=3,padding="same",downsample=True,norm=False)
            )
            prev_ch = hidden_ch
        
        '''Decoder (Upsample)'''
        for i in range(6):
            blocks.append(
                UpsampleBlock(in_ch=hidden_ch ,out_ch=hidden_ch,num_classes=num_classes,kernel=3,padding="same",upsample=True,norm=False),
            )
            blocks.append(
                UpsampleBlock(in_ch=hidden_ch*2,out_ch=hidden_ch,num_classes=num_classes,kernel=3,padding="same",upsample=False,norm=True),
            )

        self.blocks = nn.ModuleList(blocks)

        self.last = nn.Sequential(
            nn.Conv1d(hidden_ch,feature_dim,kernel_size=5,padding="same"),
            nn.Sigmoid()
        )

    

    def forward(self,x,timestep,label): # Pytorch (N,C,L)

        _x = x
        res = []
        # Encoding
        cnt = 0
        while cnt < 6:
            _x = self.blocks[cnt*2](_x,timestep,label)
            res.append(_x)
            _x = self.blocks[cnt*2 + 1](_x,timestep,label) # downsample
            cnt += 1
        
        # Decoding
        res_cnt = 5
        while cnt < 12:
            _x = self.blocks[cnt*2](_x,timestep,label) # upsample
            _x = torch.concat([_x,res[res_cnt]],dim=1)
            _x = self.blocks[cnt*2 + 1](_x,timestep,label)
            res_cnt -= 1
            cnt += 1
        
        _x = self.last(_x)
        return _x


        

            



    

