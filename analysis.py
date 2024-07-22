import argparse
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
import os

from models.generative import eeg_cgan
from utils import load_config
from model_utils import generate_samples
from utils import load_numpy_data
from analysis_utils import plot_pca, plot_tsne

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--ckpt',type=str)
parser.add_argument('--config',type=str)
parser.add_argument('--save',type=str)


args = parser.parse_args()
config = load_config(args.config)

generator = eeg_cgan.Generator(**config.get('generator',dict()))

(train_data,train_label) , (test_data,test_label) = load_numpy_data(args.data)

ckpt = torch.load(os.path.join(args.ckpt,'checkpoint.pth'),map_location=torch.device('cpu'))
generator.load_state_dict(ckpt['gen_state_dcit'])


samples,labels = generate_samples(generator,num_samples=1000,sample_per_batch=10)
print(samples.shape,labels.shape)

save_path = os.path.join(args.save,Path(args.ckpt).stem)
os.makedirs(save_path,exist_ok=True)

np.save(os.path.join(save_path,'synthesize.npy'), {'data':samples,'labels':labels})

plot_pca(real=train_data,fake=samples,save_path=save_path)
plot_tsne(real=train_data,fake=samples,save_path=save_path)













