import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import argparse

def visualize_ts_lineplot(
    X,
    Y,
    num:int = 3,
    legend_fontsize:int = 12,
    tick_size:int = 10,
    path='/visualize/sample.png'): 
    
    assert len(X.shape) == 3

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
    plt.close(fig)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str)
    parser.add_argument('--save',type=str)

    args = parser.parse_args()

    data = np.load(args.data,allow_pickle=True).item()
    visualize_ts_lineplot(data['data'],data['labels'],path=os.path.join(args.save,Path(args.data).stem+".png"))

