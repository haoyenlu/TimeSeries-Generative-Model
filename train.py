import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np

from utils import load_config
from dataset import ULF_Classification_Dataset, ULF_Generative_Dataset
from model_utils import  get_trainer_from_config
from train_utils import LinearLrDecay

from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample, plot_confusion_matrix

from argument import train_argument, train_classification_argument
from data_utils import FeatureWiseScaler



def train_generative():
    args = train_argument()
    config = load_config(args.config)

    curr_date = datetime.now().strftime("%d%m%Y_%H%M%S")


    
    # folder path
    checkpoint_path = os.path.join(args.ckpt,curr_date) # checkpoint
    log = os.path.join(args.log,curr_date) # log
    output = os.path.join(args.save,curr_date) # save folder path

    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(log,exist_ok=True)
    os.makedirs(output,exist_ok=True)

    # file name
    initial_setting = "initial_setting.pth"
    best_weight = "best_weight.pth"


    # load ulf data
    data = np.load(args.data, allow_pickle=True).item()

    # save the initial setting - reused trainer for training other task
    trainer = get_trainer_from_config(args,config)
    trainer.save_weight(os.path.join(checkpoint_path,initial_setting))

    for task in args.task:
        print(f"---- Training on {task} ------")

        # Load Data
        train_dataset = ULF_Generative_Dataset(data[task])
        train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True) 
        

        # logger
        if args.log is not None:
            os.makedirs(os.path.join(log,task),exist_ok=True)
            writer = SummaryWriter(log)
        else:
            writer = None

        # load initial setting before training - load weight before instantiate scheduler
        trainer.load_weight(os.path.join(checkpoint_path,initial_setting))

        # Learning rate scheduler
        scheduler = LinearLrDecay(trainer.optimizer,config['optimizer']['lr'],0.0,0,args.max_iter)


        # training
        os.makedirs(os.path.join(checkpoint_path,task),exist_ok=True)
        trainer.train(train_dataloader,args.max_iter,args.save_iter,scheduler,writer,verbal = args.verbal, save_path=os.path.join(checkpoint_path,task,best_weight))


        # generate samples
        samples = trainer.generate_samples(num_samples=100,num_per_batch=10)

        save_folder = os.path.join(output,task)
        os.makedirs(save_folder,exist_ok=True)
        np.save(os.path.join(save_folder,'synthesize.npy'), samples)

        # analyze with pca and tsne
        train_data = train_dataset._get_numpy()

        plot_pca(real=train_data,fake=samples,save_path=save_folder)
        plot_tsne(real=train_data,fake=samples,save_path=save_folder)
        plot_umap(real=train_data,fake=samples,save_path=save_folder)
        plot_sample(real=train_data,fake=samples,save_path=save_folder)



def train_classification():
    args = train_classification_argument()
    config = load_config(args.config)

    curr_date = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    # folder path
    checkpoint_path = os.path.join(args.ckpt,curr_date) # checkpoint
    log = os.path.join(args.log,curr_date) # log
    output = os.path.join(args.save,curr_date) # save folder path

    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(log,exist_ok=True)
    os.makedirs(output,exist_ok=True)

    # logger
    if args.log is not None:
        os.makedirs(log,exist_ok=True)
        writer = SummaryWriter(log)
    else:
        writer = None

    best_original_weight = 'best_original_weight.pth'
    best_augmentation_weight = 'best_augmentation_weight.pth'
    initial_weight = 'initial_weight.pth'

    train_data, train_label = preprocess_data(args.train_data)
    test_data, test_label = preprocess_data(args.test_data)
    
    
    train_dataset = ULF_Classification_Dataset(train_data,train_label)
    test_dataset = ULF_Classification_Dataset(test_data,test_label)
    train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True)
    test_dataloader = DataLoader(test_dataset,config['batch_size'],shuffle=True)



    '''Train without augmentation'''
    trainer = get_trainer_from_config(args,config)
    trainer.save_weight(os.path.join(checkpoint_path,initial_weight))
    trainer.train(train_dataloader,test_dataloader,args.max_iter,writer,os.path.join(checkpoint_path,best_original_weight))
    trainer.load_weight(os.path.join(checkpoint_path,best_original_weight))
    predictions = trainer.make_prediction(test_data)
    plot_confusion_matrix(test_label,predictions,output,title="Original_Prediction")


    '''Train with augmentation'''
    if args.aug_data is not None:
        aug_data, aug_label = preprocess_data(args.aug_data,scale=False)
        train_data = np.concatenate([train_data,aug_data],axis=0)
        train_label = np.concatenate([train_label,aug_label],axis=0)

        train_dataset = ULF_Classification_Dataset(train_data,train_label)
        train_dataloader = DataLoader(train_dataset,config['batch_size'],shuffle=True)
        trainer.load_weight(os.path.join(checkpoint_path,initial_weight)) # load initial weight
        trainer.train(train_dataloader,test_dataloader,args.max_iter,writer,os.path.join(checkpoint_path,best_augmentation_weight))
        trainer.load_weight(os.path.join(checkpoint_path,best_augmentation_weight))
        predictions = trainer.make_prediction(test_data)
        plot_confusion_matrix(test_label,predictions,output,title="Augmented_Prediction")






def preprocess_data(data_path,scale=True):
    data_dict = np.load(data_path,allow_pickle=True).item()
    tasks = np.array(list(data_dict.keys()))
    data = []
    label = []

    scaler = FeatureWiseScaler((0,1))

    for key, value in data_dict.items():
        np_value = np.array(value)
        B , T, C = np_value.shape
        if scale: np_value = scaler.fit_transform(np_value)
        data.append(np_value)
        l = np.argwhere(tasks == key)
        label.append([l] * B)
    
    data = np.concatenate(data)
    label = np.concatenate(label).squeeze()

    return data, label




if __name__ == '__main__':
    train_classification()