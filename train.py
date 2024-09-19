from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

from utils import load_config
from dataset import ULF_Classification_Dataset, ULF_Generative_Dataset
from model_utils import  get_trainer_from_config
from train_utils import LinearLrDecay

from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample, plot_confusion_matrix

from argument import train_argument, train_classification_argument
from data_utils import FeatureWiseScaler

from tqdm import tqdm


def train_generative_model(config,data,max_iter,save_iter,verbal,ckpt_dir):
    os.makedirs(ckpt_dir,exist_ok=True)
    trainer = get_trainer_from_config(config)
    # scale data
    scaler = FeatureWiseScaler(feature_range=(0,1))
    data = scaler.fit_transform(data)

    scheduler = LinearLrDecay(trainer.optimizer,config['optimizer']['lr'],0.0,0,max_iter)
    dataset = ULF_Generative_Dataset(data)
    dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True)
    trainer.train(dataloader,max_iter,save_iter,scheduler,writer=None,verbal=verbal,save_path=os.path.join(ckpt_dir,'best_weight.pth'))
    samples = trainer.generate_samples(num_samples=data.shape[0],num_per_batch=10)

    return dataset._getdata_() , samples


def train_classificaton_model(config,train_data,train_label,test_data,test_label,max_iter,verbal,ckpt_dir,aug_data=None, aug_label=None):
    os.makedirs(ckpt_dir,exist_ok=True)
    trainer = get_trainer_from_config(config)
    # scale data
    scaler = FeatureWiseScaler(feature_range=(0,1))
    scaler.fit(np.concatenate([train_data,test_data],axis=0))
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    if aug_data and aug_label:
        train_data = np.concatenate([train_data, aug_data],axis=0)
        train_label = np.concatenate([train_label,aug_label],axis=0)
        
    train_dataset = ULF_Classification_Dataset(train_data,train_label)
    train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    test_dataset = ULF_Classification_Dataset(test_data,test_label)
    test_dataloader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=True)
    trainer.train(train_dataloader,test_dataloader,max_iter,os.path.join(ckpt_dir,'best_weight.pth'),verbal=verbal,writer=None)
    trainer.load_weight(os.path.join(ckpt_dir,'best_weight.pth'))
    prediction = trainer.make_prediction(test_data)

    return prediction