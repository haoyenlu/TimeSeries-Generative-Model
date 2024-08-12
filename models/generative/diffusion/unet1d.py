import torch
import torch.nn as nn


from models.generative.diffusion.model_utils import AdaInsNorm


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
    
    def forward(self,x,timestep):
        if self.norm:
            x = self.ln(x,timestep)

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
        # blocks.append(nn.BatchNorm1d(out_ch))
        blocks.append(nn.LeakyReLU(0.2))

        self.blocks =  nn.Sequential(*blocks)

    def forward(self,x,timestep):
        if self.norm:
            x = self.ln(x,timestep)

        x = self.blocks(x)
        return x


class Unet1D(nn.Module):
    def __init__(self,seq_len,feature_dim,num_classes,hidden_ch,emb_dim,kernel_size):
        super(Unet1D,self).__init__()

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_ch
        self.emb_dim = emb_dim
        self.kernel_size = kernel_size
        self.depth = len(hidden_ch)

        blocks = []
        
        '''Encoder (Downsample)'''
        prev_ch = feature_dim
        for i in range(self.depth):
            blocks.append(
                DownSampleBlock(in_ch=prev_ch,out_ch=hidden_ch[i],num_classes=num_classes,kernel=kernel_size,padding="same",downsample=False,norm=True),
            )
            blocks.append(
                DownSampleBlock(in_ch=hidden_ch[i],out_ch=hidden_ch[i],num_classes=num_classes,kernel=kernel_size,padding="same",downsample=True,norm=False)
            )
            prev_ch = hidden_ch[i]

        self.down_blocks = nn.ModuleList(blocks)


        blocks = []
        
        '''Decoder (Upsample)'''
        prev_ch = hidden_ch[-1]
        for i in range(self.depth-1,-1,-1):
            blocks.append(
                UpsampleBlock(in_ch=prev_ch ,out_ch=hidden_ch[i],num_classes=num_classes,kernel=kernel_size,padding="same",upsample=True,norm=False),
            )
            blocks.append(
                UpsampleBlock(in_ch=hidden_ch[i]*2,out_ch=hidden_ch[i],num_classes=num_classes,kernel=kernel_size,padding="same",upsample=False,norm=True),
            )
            prev_ch = hidden_ch[i]

        self.up_blocks = nn.ModuleList(blocks)

        self.last = nn.Sequential(
            nn.Conv1d(hidden_ch[0],feature_dim,kernel_size=kernel_size,padding="same"),
            nn.Sigmoid()
        )

    

    def forward(self,x,timestep,label=None): # Pytorch (N,C,L)

        _x = x.transpose(1,2)
        res = []
        # Encoding
        cnt = 0
        while cnt < self.depth:
            _x = self.down_blocks[cnt*2](_x,timestep) # residual
            res.append(_x)
            _x = self.down_blocks[cnt*2 + 1](_x,timestep) # downsample
            cnt += 1
        

        # Decoding
        res_cnt = self.depth -1
        cnt = 0
        while cnt < self.depth:
            _x = self.up_blocks[cnt*2](_x,timestep) # upsample
            _x = torch.concat([_x,res[res_cnt]],dim=1)
            _x = self.up_blocks[cnt*2 + 1](_x,timestep) # residual
            res_cnt -= 1
            cnt += 1
        
        _x = self.last(_x)
        return _x.transpose(1,2)


        

            



    

