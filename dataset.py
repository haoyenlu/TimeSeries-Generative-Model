from torch.utils.data import Dataset
import numpy as np
from data_utils import FeatureWiseScaler

class ULF_Generative_Dataset(Dataset):
    def __init__(self,data,scale_range=(0,1)):
        super(ULF_Generative_Dataset,self).__init__()


        '''Only given specific task data'''
        train_data = np.array(data)
        
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


class ULF_Classification_Dataset(Dataset):
    def __init__(self,data,label):
        super(ULF_Classification_Dataset,self).__init__()
        assert data.shape[0] == label.shape[0]
        
        self.data = data
        self.label = label


    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index,:,:] , self.label[index]


    
