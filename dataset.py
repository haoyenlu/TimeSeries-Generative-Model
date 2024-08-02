from torch.utils.data import Dataset
import numpy as np
from data_utils import FeatureWiseScaler

class UpperLimbMotionDataset(Dataset):
    def __init__(self,data,task='T01',labels=None,scale_range=(0,1)):
        super(UpperLimbMotionDataset,self).__init__()

        
        self.data = np.load(data,allow_pickle=True).item()


        '''Only given specific task data'''
        train_data = np.concatenate(self.data[task],axis=0)
        
        print(train_data.shape)
        
        '''Scaling'''
        scaler = FeatureWiseScaler(scale_range)
        scaler.fit(train_data)
        self.train = scaler.transform(train_data).astype(np.float32)

        self.labels=labels
        

            
    
    def __len__(self) -> int:
        return self.train.shape[0]
        
        

    def __getitem__(self, index):
        if self.labels is not None:
            return self.train[index,:,:] , self.labels[index,:]
        else:
            return self.train[index,:,:], None



    
