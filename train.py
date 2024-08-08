import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import numpy as np

from utils import load_config
from dataset import UpperLimbMotionDataset
from model_utils import  get_trainer_from_config
from train_utils import LinearLrDecay

from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample

from argument import train_argument, train_classification_argument







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
        train_dataset = UpperLimbMotionDataset(data[task])
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

    # curr_date = datetime.now().strftime("%d%m%Y_%H%M%S")


    
    # # folder path
    # checkpoint_path = os.path.join(args.ckpt,curr_date) # checkpoint
    # log = os.path.join(args.log,curr_date) # log
    # output = os.path.join(args.save,curr_date) # save folder path

    # os.makedirs(checkpoint_path,exist_ok=True)
    # os.makedirs(log,exist_ok=True)
    # os.makedirs(output,exist_ok=True)


    # best_weight = 'best_weight.pth'

    '''TODO: preprocess the label '''
    train_data, train_label = preprocess_data(args.train_data)




def preprocess_data(data_path):
    data_dict = np.load(data_path,allow_pickle=True).item()
    tasks = np.array(list(data_dict.keys()))
    data = []
    label = []

    for key, value in data_dict.items():
        np_value = np.array(value)
        B , T, C = np_value.shape
        data.append(np_value)
        l = np.argwhere(tasks == key)
        label.append([l] * B)
    
    data = np.concatenate(data)
    label = np.concatenate(label).squeeze()

    print(data.shape)
    print(label.shape)
    return data, label




if __name__ == '__main__':
    train_classification()