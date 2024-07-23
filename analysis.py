import argparse
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
import os

# from models.generative.gan import eeg_cgan
from models.generative.diffusion.diffusion_ts import Diffusion
from models.generative.diffusion.unet1d import Unet1D

from utils import load_config
from model_utils import generate_samples_diffusion , generate_samples_gan
from utils import load_numpy_data
from analysis_utils import plot_pca, plot_tsne, plot_umap

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--ckpt',type=str)
parser.add_argument('--config',type=str)
parser.add_argument('--save',type=str)


args = parser.parse_args()
config = load_config(args.config)



(train_data,train_label) , (test_data,test_label) = load_numpy_data(args.data)

ckpt = torch.load(os.path.join(args.ckpt,'checkpoint.pth'),map_location=torch.device('cpu'))

'''GAN model'''
# generator = eeg_cgan.Generator(**config.get('generator',dict()))
# generator.load_state_dict(ckpt['gen_state_dict'])
# samples,labels = generate_samples(generator,num_samples=1000,sample_per_batch=10)

'''Diffusion model'''
backbone = Unet1D(**config.get('backbone',dict()))
diffusion_model = Diffusion(backbone,**config.get('diffusion',dict()))
diffusion_model.load_state_dict(ckpt['model'])

samples , labels = generate_samples_diffusion(diffusion_model,num_samples=1000,sample_per_batch=10)


print(samples.shape,labels.shape)

save_path = os.path.join(args.save,Path(args.ckpt).stem)
os.makedirs(save_path,exist_ok=True)

np.save(os.path.join(save_path,'synthesize.npy'), {'data':samples,'labels':labels})

plot_pca(real=train_data,fake=samples,save_path=save_path)
plot_tsne(real=train_data,fake=samples,save_path=save_path)
plot_umap(real=train_data,fake=samples,save_path=save_path)













