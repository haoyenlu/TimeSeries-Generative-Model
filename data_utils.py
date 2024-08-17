import numpy as np
from typing import Optional , Tuple, Union
from dtaidistance import dtw_barycenter

import logging


logger = logging.getLogger("augmentation")
logger.setLevel(logging.DEBUG)



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
    


class DTWBarycentricAveraging:
    def __init__(self):
        super(DTWBarycentricAveraging, self).__init__()

    def generate(
        self,
        X: np.ndarray,
        n_samples: int = 1,
        **kwargs,
    ) -> np.ndarray :

        # Draw random sample from the dataset
        random_samples = np.random.choices(range(X.shape[0]), k=n_samples)
        initial_timeseries = X[random_samples]

        self._dtwba(
                X_subset=X,
                n_samples=n_samples,
                initial_timeseries=initial_timeseries,
                **kwargs,
            )

    def _dtwba(
        self,
        X_subset: np.ndarray,
        n_samples: int,
        initial_timeseries: Optional[np.ndarray],
        **kwargs,
    ) -> np.ndarray:
        samples = []
        for i, st in enumerate(initial_timeseries):
            samples.append(
                dtw_barycenter.dba(
                    s=X_subset,
                    c=st,
                    nb_initial_samples=n_samples,
                    **kwargs,
                )
            )
        return np.array(samples)