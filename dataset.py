from torch.utils.data import Dataset
import numpy as np
from data_utils import FeatureWiseScaler

class UpperLimbMotionDataset(Dataset):
    def __init__(self,task_data,test_task_data,task='T01',labels=None,scale_range=(0,1),mode='train'):
        super(UpperLimbMotionDataset,self).__init__()

        assert mode in ['train','test'],"Only support train and test mode"
        self.mode = mode
        
        '''Only given specific task data'''
        train_data = np.concatenate(task_data[task],axis=0)
        test_data = np.concatenate(test_task_data[task],axis=0)

        
        '''Scaling'''
        scaler = FeatureWiseScaler(scale_range)
        scaler.fit(np.concatenate([train_data,test_data],axis=0))
        self.train = scaler.transform(train_data).astype(np.float32)
        self.test = scaler.transform(test_data).astype(np.float32)

        self.labels=labels
        

            
    
    def __len__(self) -> int:
        # Train
        if self.mode == 'train':
            return self.train.shape[0]
        
        # Test
        else:
            return self.test.shape[0]
        

    def __getitem__(self, index):
        # Train
        if self.mode =='train':
            if self.labels is not None:
                return self.train[index,:,:] , self.labels[index,:]
            else:
                return self.train[index,:,:], None
            
        # Test
        else:
            if self.labels is not None:
                return self.test[index,:,:] , self.labels[index,:]
            else:
                return self.test[index,:,:], None
    


    
