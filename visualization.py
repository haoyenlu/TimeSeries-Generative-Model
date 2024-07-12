import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_ts_lineplot(
    X,
    Y,
    num:int = 5,
    legend_fontsize:int = 12,
    tick_size:int = 10,
    path='/visualize/sample.png'): 
    
    assert len(X.shape) == 3
    os.makedirs(path,exist_ok=True)

    fig , axs = plt.subplots(num,1,figsize=(14,10))
    if num == 1: axs = [axs]

    ids = np.random.choice(X.shape[0],size=num,replace=False)
    for i , sample_id in enumerate(ids):
        for feat_id in range(X.shape[2]):
            sns.lineplot(x=range(X.shape[1]),y = X[sample_id,:,feat_id],ax=axs[i])
    
        if Y is not None:
            axs[i].tick_params(labelsize=tick_size,which='both')
            axs[i].set_title(Y[sample_id],fontsize=legend_fontsize)


    plt.savefig(path)