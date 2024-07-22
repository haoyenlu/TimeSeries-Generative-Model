import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import os


def plot_pca(real,fake,save_path='./save'):
  _ , feats , seq_len = real.shape
  pca = PCA(2)
  real_transform = pca.fit_transform(real.reshape(-1,seq_len * feats))
  fake_transform = pca.transform(fake.reshape(-1,seq_len * feats))
  plt.scatter(real_transform[:,0],real_transform[:,1],label="Real",s=5)
  plt.scatter(fake_transform[:,0],fake_transform[:,1],label="Fake",s=5)
  plt.legend()
  plt.savefig(os.path.join(save_path,"PCA.png"))



def plot_tsne(real,fake,save_path='./save'):
  sample , feats , seq_len = real.shape
  X = np.concatenate([real.reshape(-1,feats*seq_len),fake.reshape(-1,feats*seq_len)])
  Y = np.concatenate([np.ones(real.shape[0]),np.zeros(fake.shape[0])])
  X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=15).fit_transform(X)
  plt.scatter(X_embedded[np.argwhere(Y == 1),0],X_embedded[np.argwhere(Y == 1),1],c='red',label="real",s=5)
  plt.scatter(X_embedded[np.argwhere(Y == 0),0],X_embedded[np.argwhere(Y == 0),1],c='blue',label="fake",s=5)
  plt.title("t-SNE Distribution")
  plt.legend()
  plt.savefig(os.path.join(save_path,"TSNE.png"))
