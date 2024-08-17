import numpy as np
import random


class FeatureWiseScaler:
    def __init__(self,feature_range = (0,1)):
        # assert len(feature_range) == 2
        self.EPS = 1e-18
        self.min_v, self.max_v = feature_range
    
    def fit(self,X):
        '''Shape: (B,T,C)'''
        channels = X.shape[2]
        self.mins = np.zeros(channels)
        self.maxs = np.zeros(channels)

        for i in range(channels):
            self.mins[i] = np.min(X[:,:,i])
            self.maxs[i] = np.max(X[:,:,i])

        return self

    def transform(self,X):
        return ((X-self.mins) / (self.maxs - self.mins + self.EPS)) * (self.max_v - self.min_v) + self.min_v
    

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    

class TimeWarping:
    def __init__(self,num_operation=10,warp_factor=0.2):
        self.num_operation = num_operation
        self.warp_factor = warp_factor

    
    def generate(self,X: np.ndarray):
        B, T, C = X.shape

        warped_series = X.copy()
        for i in range(B):
            for j in range(C):
                for _ in range(self.num_operation):
                    operation_type = random.choice(['insert','delete'])
                    index = random.randint(1,T-2)
                    if operation_type == 'insert':
                        insertion_value = (warped_series[i,index - 1,j] + warped_series[i,index,j]) * 0.5
                        warp_amount = insertion_value * self.warp_factor * random.uniform(-1,1)
                        warped_series[i,:,j] = np.insert(warped_series[i,:,j],index,insertion_value + warp_amount)
                    
                    elif operation_type == 'delete':
                        warped_series[i,:,j] = np.delete(warped_series[i,:,j],index)
                    else:
                        raise ValueError('Invalid operation type')
        return warped_series