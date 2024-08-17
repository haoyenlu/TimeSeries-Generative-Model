import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import umap
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def plot_pca(real,fake,save_path='./save'):
  print('Geneating PCA plot')
  _ , seq_len , feats = real.shape
  pca = PCA(2)
  real_transform = pca.fit_transform(real.reshape(-1,seq_len * feats))
  fake_transform = pca.transform(fake.reshape(-1,seq_len * feats))
  plt.scatter(real_transform[:,0],real_transform[:,1],c='red',label="Real",s=5)
  plt.scatter(fake_transform[:,0],fake_transform[:,1],c='blue',label="Fake",s=5)
  plt.legend()
  plt.savefig(os.path.join(save_path,"PCA.png"))
  plt.close()



def plot_tsne(real,fake,save_path='./save'):
  print('Generating TSNE plot')
  _ , seq_len , feats = real.shape
  X = np.concatenate([real.reshape(-1,feats*seq_len),fake.reshape(-1,feats*seq_len)])
  Y = np.concatenate([np.ones(real.shape[0]),np.zeros(fake.shape[0])])
  X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=15).fit_transform(X)
  plt.scatter(X_embedded[:real.shape[0],0],X_embedded[:real.shape[0],1],c='red',label="real",s=5)
  plt.scatter(X_embedded[real.shape[0]:,0],X_embedded[real.shape[0]:,1],c='blue',label="fake",s=5)
  plt.title("t-SNE Distribution")
  plt.legend()
  plt.savefig(os.path.join(save_path,"TSNE.png"))
  plt.close()


def plot_umap(real,fake,save_path='./save'):
  print('Generating UMAP plot')
  reducer = umap.UMAP()
  _ , seq_len , feats = real.shape
  X = np.concatenate([real.reshape(-1,feats * seq_len),fake.reshape(-1,feats* seq_len)])
  Y = np.concatenate([np.ones(real.shape[0]),np.zeros(fake.shape[0])])
  X_embed = reducer.fit_transform(X)
  plt.scatter(X_embed[:real.shape[0],0],X_embed[:real.shape[0],1],c='red',label="real",s=5)
  plt.scatter(X_embed[real.shape[0]:,0],X_embed[real.shape[0]:,1],c='blue',label="fake",s=5)
  plt.title("Umap Distribution")
  plt.legend()
  plt.savefig(os.path.join(save_path,"umap.png"))
  plt.close()


def plot_sample(real,fake,save_path='./save'):
  real_num , seq_len , feats = real.shape
  fake_num , seq_len , feats = fake.shape

  num = min(real_num, fake_num)

  index = np.random.randint(0,num,size=3)


  fig, axs = plt.subplots(3,2,figsize=(8,6))
  for i in range(3):
    for f in range(feats):
      axs[i,0].plot(real[index[i],:,f])
      axs[i,1].plot(fake[index[i],:,f])
  
  axs[0,0].set_title("Real")
  axs[0,1].set_title("Fake")

  plt.savefig(os.path.join(save_path,"sample.png"))
  plt.close()





def plot_confusion_matrix(real,prediction,save_path='./save',title="Prediction"):
    print('Generating confusion matrix plot')
    fig = plt.figure(figsize = (12,12))
    acc = accuracy_score(real,prediction)
    cm = confusion_matrix(real, prediction)
    f = sns.heatmap(cm, annot=True, fmt='d')
    f.figure.suptitle(f"{title} | ACC:{acc*100:.2f}%")
    f.figure.savefig(os.path.join(save_path,f"{title}.png"))
    plt.close(fig)