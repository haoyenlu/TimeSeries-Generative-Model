import torch
import torch.nn as nn



class Unet1D(nn.Module):
    def __init__(self,seq_len,feature_dim,num_classes,hidden_dim,label_emb_dim):
        super(Unet1D,self).__init__()

        