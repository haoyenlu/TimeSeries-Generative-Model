import torch
import torch.nn as nn


class InceptionModule(nn.Module):
  def __init__(self,input_dim,filter_size=32,kernels=[10,20,40],use_bottleneck=True):
    super(InceptionModule,self).__init__()
    self.bottleneck_size = filter_size
    self.use_bottleneck = use_bottleneck
    self.filter_size = filter_size
    self.input_inception = nn.Conv1d(input_dim,self.bottleneck_size,kernel_size=1,padding='same',bias=False)

    self.conv_list = []
    prev = input_dim if not use_bottleneck else self.bottleneck_size

    for kernel in kernels:
      self.conv_list.append(nn.Conv1d(prev,filter_size,kernel_size=kernel,padding='same',bias=False))

    self.conv_list = nn.ModuleList(self.conv_list)

    self.max_pool_1 = nn.MaxPool1d(kernel_size=3,padding=1,stride=1)
    self.conv6 = nn.Conv1d(input_dim,self.filter_size,kernel_size=1,padding='same',bias=False)

    # self.lstm = nn.LSTM(input_dim,filter_size,num_layers=2,batch_first=True)

    self.bn = nn.BatchNorm1d((len(kernels) + 1) * filter_size)
    self.act = nn.GELU()


  def forward(self,x): # NCL
    _x = x
    if self.use_bottleneck:
      x = self.input_inception(x)

    x_list = []
    for conv in self.conv_list:
      x_list.append(conv(x))

    # lstm_x, (_,_) = self.lstm(_x.permute((0,2,1)))
    # lstm_x = lstm_x.permute((0,2,1))

    _x = self.max_pool_1(_x)
    x_list.append(self.conv6(_x))

    # x_list.append(lstm_x)

    x = torch.concat(x_list,dim=1) 
    x = self.bn(x)
    x = self.act(x)

    return x

class ResidualLayer(nn.Sequential):
  def __init__(self,input_dim,output_dim):
    super(ResidualLayer,self).__init__()
    self.conv = nn.Conv1d(input_dim,output_dim,kernel_size=1,padding='same',bias=False)
    self.bn = nn.BatchNorm1d(output_dim)
    self.act = nn.ReLU()

  def forward(self,residual_input,input):
    residual = self.conv(residual_input)
    residual = self.bn(residual) 

    x = residual + input
    x = self.act(x)
    return x
  


class FCNLayer(nn.Module):
  def __init__(self,input_dim,output_dim,kernel_size,stride=1,padding=1):
    super(FCNLayer,self).__init__()
    self.model = nn.Sequential( 
      nn.Conv1d(input_dim,output_dim,kernel_size,stride=stride,padding=padding),
      nn.BatchNorm1d(output_dim),
      nn.ReLU())
  
  def forward(self,x):
    return self.model(x)




class InceptionTime(nn.Module):
    def __init__(self,sequence_len,feature_size,label_dim, 
                inception_filter=32,fcn_filter = 128,depth=6,fcn_layers = 6,kernels = [10,20,40],dropout=0.2,
                use_residual=True, use_bottleneck=True):
        
        super(InceptionTime,self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sequence_len = sequence_len
        self.feature_size = feature_size
        self.label_dim = label_dim
        self.depth = depth
        self.use_residual = use_residual
        self.filter_size = inception_filter

        self.inceptions = []
        self.shortcuts = []

        prev = feature_size
        residual_prev = prev

        for d in range(depth):
            self.inceptions.append(InceptionModule(
                prev,
                inception_filter,
                kernels,
                use_bottleneck,
            ))

            if use_residual and d % 2 == 1: 
                self.shortcuts.append(ResidualLayer(
                    input_dim = residual_prev,
                    output_dim = (len(kernels)+2) * inception_filter
                ))
                residual_prev = prev

            prev = (len(kernels) + 2) * inception_filter

        self.inceptions = nn.ModuleList(self.inceptions)
        self.shortcuts = nn.ModuleList(self.shortcuts)

        self.fcn = []
        for i in range(fcn_layers):
            self.fcn.append(FCNLayer(prev,fcn_filter,kernel_size=5,stride=2,padding=2))
            prev = fcn_filter
        
        self.fcn = nn.Sequential(*self.fcn)
        self.out = nn.Linear(fcn_filter * (sequence_len // 2 ** (fcn_layers)),label_dim)
        # self.out = nn.Linear(prev,label_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

    def forward(self,x): # input shape: (N,L,C)
        assert self.sequence_len == x.shape[1] and self.feature_size == x.shape[2]
        x = x.transpose(2,1)


        res_input = x
        s_index = 0
        for d in range(self.depth):
            x = self.inceptions[d](x)

            if self.use_residual and d % 3 == 2:
                x = self.shortcuts[s_index](res_input,x)
                res_input = x
                s_index += 1

        x = self.fcn(x)
        # x = torch.mean(x,dim=2) # NCL -> NC (average pooling)
        x = torch.flatten(x,start_dim=1)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)

        return x
  
