import numpy as np
import argparse
import os
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_utils import  get_trainer_from_config
from utils import load_config
from data_utils import  WindowWarping, MovingAverageFilter
from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample, plot_confusion_matrix
from train import train_generative_model, train_classificaton_model

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from logger import Logger


logger = Logger()
curr_date = datetime.now().strftime("%d%m%Y_%H%M%S")

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--test_patient','-tp',nargs='+',type=str)
parser.add_argument('--ih',action='store_true',help="Include Healthy Patient")
parser.add_argument('--cc',type=str, help="Classification Model Config") 
parser.add_argument('--gc',type=str, help="Generative Model Config",default=None)
parser.add_argument('--max_ci',type=int, help="Max Classification Model Training Iteration")
parser.add_argument('--max_gi',type=int, help="Max Generative Model Training Iteration")
parser.add_argument('--save_gi',type=int, help="Generative Model Save Iteration")
parser.add_argument('--output','-o',type=str)
parser.add_argument('--ckpt',type=str)
parser.add_argument('--log',type=str)
parser.add_argument('--verbal',action='store_true')
parser.add_argument('--maw',help="Moving Average Window Size",type=int, default=0)
parser.add_argument('--ratio',type=float,help="The ratio of augmentation data size",default=1)

args = parser.parse_args()


# DIRECTORY
output_dir = os.path.join(args.output, curr_date)
os.makedirs(output_dir)
ckpt_dir = os.path.join(args.ckpt, curr_date)
os.makedirs(ckpt_dir,exist_ok=True)
# log_dir = os.path.join(args.log, curr_date)
# os.makedirs(log_dir,exist_ok=True)


# CONFIG
cc_config = load_config(args.cc)
gc_config = load_config(args.gc)


# logger
# writer = SummaryWriter(log_dir)  
writer = None


def make_dataset_and_labels(data: dict, tasks: list):
    dataset = []
    labels = []

    np_tasks = np.array(tasks)
    
    for key,value in data.items():
        label = np.argwhere(np_tasks == key)
        dataset.append(value)
        labels.append([label] * len(value))
    
    dataset = np.concatenate(dataset,axis=0)
    labels = np.concatenate(labels,axis=0).squeeze()
    
    assert dataset.shape[0] == labels.shape[0]
    return dataset ,labels


def main(TEST_PATIENT: str):
    logger.info(f"Processing Data. Leave {TEST_PATIENT} out. ")
    data = np.load(args.data,allow_pickle=True).item()
    train_dataset = dict()
    test_dataset = dict()

    for type, type_dict in data.items():
        logger.info(f"Processing Data type {type}.")
        train_dataset[type] = defaultdict(list)
        test_dataset[type] = defaultdict(list)

        for patient , patient_dict in type_dict.items():
            for task , task_data in patient_dict.items():
                if patient == TEST_PATIENT: 
                    test_dataset[type][task].append(task_data)
                else: 
                    train_dataset[type][task].append(task_data)

    # TODO: Data augmentation and Preprocessing
    tasks = list(train_dataset['Strokes'].keys())
    filter = MovingAverageFilter(window_size=args.maw) if args.maw != 0 else None
    AUG_data = dict()


    # Diffusion Data augmentation
    with tqdm(total=len(train_dataset.keys())) as pbar:
        for task in tasks:              
            # only use stroke data for diffusion augmentation
            if len(train_dataset['Strokes'][task]) == 0:
                logger.warning(f"No Strokes data for task {task}")
                continue

            train_dataset['Strokes'][task] = np.concatenate(train_dataset['Strokes'][task],axis=0)
            train_dataset['Healthies'][task] = np.concatenate(train_dataset['Healthies'][task],axis=0)
            
            if task in test_dataset['Strokes']:
                test_dataset['Strokes'][task] = np.concatenate(test_dataset['Strokes'][task],axis=0)
            
            train_data = train_dataset['Strokes'][task]

            # Train generative model on train_data for augmentation
            logger.info(f"Training Generative Model on {task}")
            real , samples = train_generative_model(gc_config,train_data,args.max_gi,args.save_gi,args.verbal,ckpt_dir=os.path.join(ckpt_dir,TEST_PATIENT,task),ratio=args.ratio)

            if filter:
                samples = filter.apply(samples)
            AUG_data[task] = samples

            patient_output_dir = os.path.join(output_dir,TEST_PATIENT,task)
            os.makedirs(patient_output_dir,exist_ok=True)

            # visualize the augmentation
            plot_sample(real,samples,patient_output_dir)
            plot_pca(real,samples,patient_output_dir)
            plot_tsne(real,samples,patient_output_dir)
            plot_umap(real,samples,patient_output_dir)

            pbar.update(1)


    train_data, train_label = make_dataset_and_labels(train_dataset['Strokes'],tasks)
    test_data, test_label = make_dataset_and_labels(test_dataset['Strokes'],tasks)


    if args.ih:
        healthy_data, healthy_label = make_dataset_and_labels(train_dataset['Healthies'],tasks)
        train_data = np.concatenate([train_data,healthy_data],axis=0)
        train_label = np.concatenate([train_label,healthy_label],axis=0)



    logger.info("Train without Augmentation")

    prediction = train_classificaton_model(cc_config,train_data,train_label,test_data,test_label,args.max_ci,args.verbal,os.path.join(ckpt_dir,TEST_PATIENT,'Original'))
    plot_confusion_matrix(test_label, prediction, output_dir, title=f"{TEST_PATIENT}-Original-Prediction")
    orig_acc, orig_f1 = accuracy_score(test_label,prediction), f1_score(test_label,prediction,average='micro')
    logger.info(f"{TEST_PATIENT}(WITHOUT AUGMENTATION): Accuracy: {orig_acc*100:.2f}% | F1-score: {orig_f1*100:.2f}%")



    logger.info("Train with Diffusion Augmentation")

    
    train_diff_aug_data, train_diff_aug_label = make_dataset_and_labels(AUG_data,tasks)

    prediction = train_classificaton_model(cc_config,train_data,train_label,test_data,test_label,args.max_ci,args.verbal,os.path.join(ckpt_dir,TEST_PATIENT,'Diffusion-Augmentation'),aug_data=train_diff_aug_data,aug_label=train_diff_aug_label)
    plot_confusion_matrix(test_label,prediction,output_dir,title=f"{TEST_PATIENT}-Diffusion-Augmented-Prediciton")
    diffusion_aug_acc, diffusion_aug_f1 = accuracy_score(test_label,prediction), f1_score(test_label,prediction,average="micro")
    logger.info(f"{TEST_PATIENT}(WITH Diffusion AUGMENTATION): Accuracy: {diffusion_aug_acc*100:.2f}% | F1-score: {diffusion_aug_f1*100:.2f}%")




for patient in args.test_patient:
    main(patient)