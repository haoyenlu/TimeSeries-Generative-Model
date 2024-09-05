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
    

class WindowWarping:
    def __init__(self,window_ratio = 0.1,scales=[0.5,1,1.5,2]):
        self.window_ratio = window_ratio
        self.scales = scales
    
    def generate(self,X: np.ndarray):
        B, T, C = X.shape

        warped_series = X.copy()
        for i in range(B):
            warped_series[i,:,:] = self.warping_channels(warped_series[i,:,:])
            # for j in range(C):
            #     warped_series[i,:,j] = self.warping(warped_series[i,:,j])

        return warped_series
    
    def warping(self,time_series):
        n = len(time_series)
        window_size = int(n * self.window_ratio)

        # Randomly select a window start index
        start = np.random.randint(0, n - window_size)

        # Randomly select a scaling factor
        scale = np.random.choice(self.scales)

        # Warp the window
        warped_window = np.interp(np.linspace(0, window_size, int(window_size*scale)), np.arange(window_size), time_series[start: start + window_size])
        # Replace the original window with the warped window
        warped_series = np.append(time_series[:start], warped_window)
        warped_series = np.append(warped_series,time_series[start+window_size:])

        # Rescale the entire series to maintain the original length
        warped_series = np.interp(np.arange(n), np.linspace(0, n, len(warped_series)), warped_series)

        return warped_series
    
    def warping_channels(self,time_series):
        T, C = time_series.shape
        window_size = int(T * self.window_ratio)

        start = np.random.randint(0,T-window_size)
        scale = np.random.choice(self.scales)

        warped = time_series.copy()

        for channel in range(C):
            warped_window = np.interp(np.linspace(0,window_size,int(window_size * scale)), np.arange(window_size), time_series[start:start + window_size,channel])
            warped_series = np.append(time_series[:start,channel],warped_window)
            warped_series = np.append(warped_window,time_series[start + window_size:,channel])
            warped_series = np.interp(np.arange(T), np.linspace(0, T, len(warped_series)), warped_series)
            warped[:,channel] = warped_series
        
        return warped
    

class MovingAverageFilter:
    def __init__(self,window_size=10):
        self.window_size = window_size
        

    def apply(self,series):
        B,T,C = series.shape
        new_series = np.zeros(shape=(B,T,C))
        series = np.pad(series,(0,self.window_size),mode='constant',constant_values=(0,0))
        
        for b in range(B):
            for c in range(C):
                _temp = sum(series[b,:self.window_size,c])
                for i in range(T):
                    new_series[b,i,c] = _temp / self.window_size

                    _temp -= series[b,i,c]
                    _temp += series[b,i+self.window_size,c]
        
        return new_series[:,:T,:]