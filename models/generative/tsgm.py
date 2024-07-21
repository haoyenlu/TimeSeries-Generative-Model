import torch
import torch.nn as nn


class cGAN_Conv4Architecture:
    def __init__(self,seq_len,feature_dim,latent_dim,num_classes,label_emb_dim):
    

        class Generator(nn.Module):
            def __init__(self,seq_len,feature_dim,latent_dim,num_classes,
                         label_emb_dim,data_emb_dim):
                super(Generator,self).__init__()
                self.seq_len = seq_len
                self.feature_dim = feature_dim
                self.latent_dim = latent_dim
                self.num_classes = num_classes
                self.label_emb_dim = label_emb_dim
                self.data_emb_dim = data_emb_dim

                self.label_embedding = nn.Embedding(self.num_classes,self.label_emb_dim)

                self.l1 = nn.Linear(self.latent_dim + self.label_emb_dim,self.data_emb_dim * self.seq_len)
                self.leakyRelu = nn.LeakyReLU(0.2)
                self.module = nn.Sequential(
                    nn.ConvTranspose1d(self.data_emb_dim,128,kernel_size=5,stride=1,padding=2),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose1d(self.data_emb_dim,128,kernel_size=5,stride=1,padding=2),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose1d(self.data_emb_dim,128,kernel_size=5,stride=1,padding=2),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose1d(self.data_emb_dim,128,kernel_size=5,stride=1,padding=2),
                    nn.LeakyReLU(0.2),
                )

                self.conv1d = nn.Conv1d(128,1,8,padding="same")
                self.lstm = nn.LSTM(1,256,batch_first=True)
                self.conv1d_2 = nn.Conv1d(256,self.feature_dim,1,padding="same")


            def forward(self,noise,labels):
                '''Input Noise shape: (B , latent_dim)'''
                c = self.label_embedding(labels)
                x = torch.cat([noise,c],1)
                x = self.leakyRelu(self.l1(x))
                x = x.view(-1,self.data_emb_dim,self.seq_len)
                x = self.module(x)
                x = self.conv1d(x)
                x = self.lstm(x.permute(0,2,1))
                x = self.conv1d_2(x.permute(0,2,1))
                return x
            
        class Discriminator(nn.Module):
            def __init__(self,seq_len,feature_dim,num_classes,label_emb_dim):
                super(Discriminator,self).__init__()
                
                self.modules = nn.Sequential(
                    nn.Conv1d(feature_dim,64,3,stride=2,padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                    nn.Conv1d(64,128,3,2,1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                    nn.Conv1d(128,128,3,2,1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                    nn.Conv1d(128,128,3,2,1),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                )

                self.last_adv = nn.Linear(128,1)
                self.last_cls = nn.Linear(128,num_classes)

            def forward(self,sequence):
                x = self.modules(x)
                x = torch.mean(x,dim=1)
                out_adv = self.last_adv(x)
                out_cls = self.last_cls(x)

                return out_adv, out_cls
        






