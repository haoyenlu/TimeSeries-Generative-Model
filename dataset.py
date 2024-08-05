from torch.utils.data import Dataset
import numpy as np
from data_utils import FeatureWiseScaler

class UpperLimbMotionDataset(Dataset):
    def __init__(self,data,task='T01',scale_range=(0,1)):
        super(UpperLimbMotionDataset,self).__init__()

        
        self.data = np.load(data,allow_pickle=True).item()


        '''Only given specific task data'''
        train_data = np.array(self.data[task])
        
        print("Train data shape:",train_data.shape)
        
        '''Scaling'''
        scaler = FeatureWiseScaler(scale_range)
        scaler.fit(train_data)
        self.train = scaler.transform(train_data).astype(np.float32)

        

            
    
    def __len__(self) -> int:
        return self.train.shape[0]
        
        

    def __getitem__(self, index):
        return self.train[index,:,:]


    def _get_numpy(self):
        return self.train


    
