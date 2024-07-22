import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module): # NLC
  def __init__(self,d_model,max_len=1024):
    super(PositionalEmbedding,self).__init__()
    pe = torch.zeros(max_len,d_model).float()
    pe.require_grad = False

    position = torch.arange(0,max_len).float().unsqueeze(1)
    div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0) / d_model)).exp()

    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe',pe)

  def forward(self,x):
    return self.pe[:,:x.size(1)] + x
  
  
class LearnablePositionalEncoding(nn.Module): #NLC
    def __init__(self,d_model,dropout=0.1,max_len=1024):
        super(LearnablePositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1,max_len,d_model))
        nn.init.uniform_(self.pe,-0.02,0.02)

    def forward(self,x):
        x = x + self.pe
        return self.dropout(x)


class TokenEmbedding(nn.Module): # NLC
  def __init__(self,c_in,d_model):
    super(TokenEmbedding,self).__init__()
    padding = 1 if torch.__version__ >= '1.5.0' else 2
    self.tokenConv = nn.Conv1d(in_channels=c_in,out_channels=d_model,
                               kernel_size=1,padding='same',bias=False)
    for m in self.modules():
      if isinstance(m,nn.Conv1d):
        nn.init.kaiming_normal(m.weight,mode='fan_in',nonlinearity='leaky_relu')

  def forward(self,x):
    x = self.tokenConv(x.permute(0,2,1)).transpose(1,2)
    return x


class DataEmbedding(nn.Module):
  def __init__(self,c_in,d_model,dropout=0.1,max_len=1024):
    super(DataEmbedding,self).__init__()
    self.value_embedding = TokenEmbedding(c_in=c_in,d_model=d_model)
    self.position_embedding = PositionalEmbedding(d_model=d_model,max_len=max_len)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    x = self.value_embedding(x) + self.position_embedding(x)
    return self.dropout(x)
  

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = torch.nn.functional.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class EncoderBlock(nn.Module):
    def __init__(self,d_model,d_hidden,n_head,dropout):
        super(EncoderBlock,self).__init__()
        self.attn = FullAttention(n_embd=d_model,n_head=n_head,attn_pdrop=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.fn = nn.Sequential(
            nn.Linear(d_model,d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden,d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
       _x = x
       x = self.attn(x)
       x = self.dropout1(x)
       x = self.ln1(x + _x)

       _x = x
       x = self.fn(x)
       x = self.dropout2(x)
       x = self.ln2(x + _x)
       return x
   
class Encoder(nn.Module):
    def __init__(self,n_layers,d_model,fn_hidden,n_head,dropout):
        super(Encoder,self).__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model=d_model,
                         d_hidden=fn_hidden,
                         n_head=n_head,
                         dropout=dropout)
        for _ in range(n_layers)])
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x