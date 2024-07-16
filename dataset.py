import torch
from torch.utils.data import Dataset


class UpperLimbMotionDataset(Dataset):
    def __init__(self,data,label):
        super(UpperLimbMotionDataset,self).__init__()

        assert data.shape[0] == label.shape[0]

        self.data = data
        self.label = label
    
    def __len__(self) -> int:
        return self.data.shape[0]
    

    def __getitem__(self, index):
        return self.data[index,:,:] , self.label[index,:]
    


    
