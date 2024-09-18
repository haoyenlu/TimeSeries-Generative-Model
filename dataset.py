from torch.utils.data import Dataset
import numpy as np
from data_utils import FeatureWiseScaler

class ULF_Generative_Dataset(Dataset):
    def __init__(self,data):
        super(ULF_Generative_Dataset,self).__init__()

        self.data = data
    
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,:,:]


    def _getdata_(self):
        return self.data


class ULF_Classification_Dataset(Dataset):
    def __init__(self,data,label,scale_range=(0,1)):
        super(ULF_Classification_Dataset,self).__init__()
        assert data.shape[0] == label.shape[0]

        self.data = data
        self.label = label

    def __len__(self) -> int:
        return self.data.shape[0]
    

    def __getitem__(self, index):
        return self.data[index,:,:] , self.label[index]

    def _getdata_(self):
        return self.data
    
