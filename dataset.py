from torch.utils.data import Dataset
import numpy as np

class UpperLimbMotionDataset(Dataset):
    def __init__(self,task_data,labels=None):
        super(UpperLimbMotionDataset,self).__init__()
        
        self.data = task_data
        self.labels = labels
        
    
    def __len__(self) -> int:
        return self.data.shape[0]
    

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index,:,:] , self.labels[index,:]
        else:
            return self.data[index,:,:]
    


    
