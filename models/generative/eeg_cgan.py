import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce



class Generator(nn.Module):
    def __init__(self,sequence_len,feature_dim,num_classes,hidden_dim=50,latent_dim=200,label_emb_dim=5):
        super().__init__()
        self.sequence_len = sequence_len
        self.out_features = feature_dim
        self.hidden_dim = hidden_dim
        self.label_dim = label_emb_dim
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(num_classes,label_emb_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim + label_emb_dim,(sequence_len//64) * hidden_dim,bias=False),
            nn.LeakyReLU(0.2)
        )

        self.block1 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block2 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block3 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block4 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block5 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block6 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )


        self.last = nn.Sequential(
            nn.Conv1d(hidden_dim,feature_dim,kernel_size=5,padding="same"),
            nn.Sigmoid()
        )



    def make_conv1d_block(self,in_channel,out_channel,kernel=3,upsample=True):
        block = []

        if upsample:
            block.append(nn.Upsample(scale_factor=2))
        
        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.BatchNorm1d(out_channel))
        block.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*block)

    def forward(self,noise,fake_label):
        c = self.embedding(fake_label)
        out = torch.cat([noise,c],dim=1)
        out = self.fc1(out)
        out = torch.reshape(out,(-1,self.hidden_dim,self.sequence_len//64))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.last(out)
        out = torch.reshape(out,(-1,self.out_features,self.sequence_len))
        
        return out
    
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=100, adv_classes=1, cls_classes=10):
        super().__init__()
        self.adv_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, adv_classes)
        )
        self.cls_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, cls_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        out_adv = self.adv_head(x)
        out_cls = self.cls_head(x)
        return out_adv, out_cls
    
class Discriminator(nn.Module):
    def __init__(self,sequence_len,in_features,hidden_dim,num_classes):
        super().__init__()

        self.features = in_features

        self.first_conv = nn.Conv1d(in_features,hidden_dim,kernel_size=3,padding="same")

        self.block1 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block2 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block3 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block4 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block5 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block6 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )

        self.last = ClassificationHead(emb_size=sequence_len//64,adv_classes=1,cls_classes=num_classes)

    def make_conv1d_block(self,in_channel,out_channel,kernel=3,downsample=False):
        block = []

        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.LeakyReLU(0.2))
        if downsample:
            block.append(nn.AvgPool1d(kernel_size=3,padding=1,stride=2))

        return nn.Sequential(*block)
    
    def forward(self,x):
        _x = self.first_conv(x)
        _x = self.block1(_x)
        _x = self.block2(_x)
        _x = self.block3(_x)
        _x = self.block4(_x)
        _x = self.block5(_x)
        _x = self.block6(_x)
        out_adv, out_class = self.last(_x)
        return out_adv, out_class