import numpy as np

EPS = 1e-18


class FeatureWiseScaler:
    def __init__(self,feature_range = (0,1)):
        # assert len(feature_range) == 2

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
        return ((X-self.mins) / (self.maxs - self.mins + EPS)) * (self.max_v - self.min_v) + self.min_v