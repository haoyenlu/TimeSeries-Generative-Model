import numpy as np
import argparse
import os
from collections import defaultdict
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_utils import  get_trainer_from_config
from utils import load_config
from data_utils import FeatureWiseScaler , WindowWarping, MovingAverageFilter
from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample, plot_confusion_matrix
from dataset import ULF_Classification_Dataset, ULF_Generative_Dataset
from train_utils import LinearLrDecay

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
parser.add_argument('--gc',type=str, help="Generative Model Config")
parser.add_argument('--max_ci',type=int, help="Max Classification Model Training Iteration")
parser.add_argument('--max_gi',type=int, help="Max Generative Model Training Iteration")
parser.add_argument('--save_gi',type=int, help="Generative Model Save Iteration")
parser.add_argument('--output','-o',type=str)
parser.add_argument('--ckpt',type=str)
parser.add_argument('--log',type=str)
parser.add_argument('--verbal',action='store_true')

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

# TODO: preprocess dataset
# Data Structure - Healthy - H01 ~ H05 - TASK- data
#                - Stroke - P02 ~ P30 - TASK - data

def main(TEST_PATIENT):
    logger.info(f"Processing Data. Leave {TEST_PATIENT} out")
    data = np.load(args.data,allow_pickle=True).item()
    train_dataset = defaultdict(list)
    test_dataset = defaultdict(list)

    for type, type_dict in data.items():
        if not args.ih and type == 'Healthies': continue

        for patient , patient_dict in type_dict.items():
            for task , task_data in patient_dict.items():
                if patient == TEST_PATIENT: test_dataset[task].append(task_data)
                else: train_dataset[task].append(task_data)


    # TODO: Data augmentation and Preprocessing
    scaler = FeatureWiseScaler(feature_range=(0,1))
    augmenter = WindowWarping(window_ratio=0.4,scales=[0.1,0.5,1,1.5,2,2.5])
    filter = MovingAverageFilter(window_size=10)
    tasks = np.array(list(train_dataset.keys()))

    all_train_data = []
    all_train_data_aug = []
    all_train_data_aug_diffusion = []
    all_train_label = []
    all_train_label_aug = []
    all_train_label_aug_diffusion = []
    all_test_data = []
    all_test_label = []

    g_trainer = get_trainer_from_config(gc_config)
    g_trainer.save_weight(os.path.join(ckpt_dir,'initial_g.pth'))

    with tqdm(total=len(train_dataset.keys())) as pbar:
        for task in train_dataset.keys():
            train_data = np.concatenate(train_dataset[task],axis=0)
            train_data = scaler.fit_transform(train_data)
            train_data_aug = augmenter.generate(train_data)

            if len(test_dataset[task]) != 0:
                test_data = np.concatenate(test_dataset[task],axis=0)
                test_data = scaler.fit_transform(test_data)
            else:
                logger.warning(f"{TEST_PATIENT} has no {task} data!")
                test_data = None

            # TODO: train generative model on train_data for augmentation
            logger.info(f"Training Generative Model on {task}")
            g_trainer.load_weight(os.path.join(ckpt_dir,'initial_g.pth'))
            scheduler = LinearLrDecay(g_trainer.optimizer, gc_config['optimizer']['lr'], 0.0, 0, args.max_gi)
            train_generative_dataset = ULF_Generative_Dataset(train_data)
            train_generative_dataloader = DataLoader(train_generative_dataset) 
            g_trainer.train(train_generative_dataloader,args.max_gi,args.save_gi,scheduler,writer,verbal = args.verbal, save_path=os.path.join(ckpt_dir,TEST_PATIENT,task,'best_generative_weight.pth'))
            samples = g_trainer.generate_samples(num_samples=train_data.shape[0],num_per_batch=10)
            samples = filter.apply(samples)
            all_train_data_aug_diffusion.append(samples)

            patient_output_dir = os.path.join(output_dir,TEST_PATIENT,task)
            os.makedirs(patient_output_dir,exist_ok=True)
            # see the augmentation
            plot_sample(train_data,samples,patient_output_dir)
            plot_pca(train_data,samples,patient_output_dir)
            plot_tsne(train_data,samples,patient_output_dir)
            plot_umap(train_data,samples,patient_output_dir)

            # TODO: generate dataset with label
            label = np.argwhere(tasks == task)
            all_train_data.append(train_data)
            all_train_label.append([label] * train_data.shape[0])
            all_train_data_aug.append(train_data_aug)
            all_train_label_aug.append([label] * train_data_aug.shape[0])
            all_train_label_aug_diffusion.append([label] * samples.shape[0])

            if test_data is not None:
                all_test_data.append(test_data)
                all_test_label.append([label] * test_data.shape[0])
            pbar.update(1)

    all_train_data = np.concatenate(all_train_data,axis=0)
    all_train_label = np.squeeze(np.concatenate(all_train_label,axis=0))
    all_train_data_aug = np.concatenate(all_train_data_aug,axis=0)
    all_train_label_aug = np.squeeze(np.concatenate(all_train_label_aug,axis=0))
    all_test_data = np.concatenate(all_test_data,axis=0)
    all_test_label = np.squeeze(np.concatenate(all_test_label,axis=0))
    all_train_data_aug_diffusion = np.concatenate(all_train_data_aug_diffusion,axis=0)
    all_train_label_aug_diffusion = np.concatenate(all_train_label_aug_diffusion,axis=0).squeeze()

    print(all_train_data_aug_diffusion.shape)


    # TODO: train classification on augmentation and original dataset and test with test dataset
    train_dataset = ULF_Classification_Dataset(all_train_data,all_train_label)
    test_dataset = ULF_Classification_Dataset(all_test_data,all_test_label)
    train_dataloader = DataLoader(train_dataset,cc_config['batch_size'],shuffle=True)
    test_dataloader = DataLoader(test_dataset,cc_config['batch_size'],shuffle=True)

    # IMPORT CLASSIFICATION MODEL
    # train without augmentation
    logger.info("Train without Augmentation")
    trainer  = get_trainer_from_config(cc_config)
    trainer.save_weight(os.path.join(ckpt_dir,"initial.pth"))
    trainer.train(train_dataloader,test_dataloader,args.max_ci,os.path.join(ckpt_dir,f'{TEST_PATIENT}_best.pth'),verbal=args.verbal,writer=writer)
    trainer.load_weight(os.path.join(ckpt_dir,f'{TEST_PATIENT}_best.pth'))
    prediction = trainer.make_prediction(all_test_data)
    plot_confusion_matrix(all_test_label, prediction, output_dir, title=f"{TEST_PATIENT}-Original-Prediction")

    orig_acc, orig_f1 = accuracy_score(all_test_label,prediction), f1_score(all_test_label,prediction,average='micro')
    logger.info(f"{TEST_PATIENT}(WITHOUT AUGMENTATION): Accuracy: {orig_acc*100:.2f}% | F1-score: {orig_f1*100:.2f}%")

    # train with augmentation
    logger.info("Train with Time Warping Augmentation")
    train_dataset = ULF_Classification_Dataset(np.concatenate([all_train_data,all_train_data_aug],axis=0),np.concatenate([all_train_label,all_train_label_aug],axis=0))
    train_dataloader = DataLoader(train_dataset,cc_config['batch_size'],shuffle=True)
    trainer.load_weight(os.path.join(ckpt_dir,"initial.pth"))
    trainer.train(train_dataloader,test_dataloader,args.max_ci,os.path.join(ckpt_dir,f'{TEST_PATIENT}_best_aug.pth'),verbal=args.verbal,writer=writer)
    trainer.load_weight(os.path.join(ckpt_dir,f'{TEST_PATIENT}_best_aug.pth'))
    prediction = trainer.make_prediction(all_test_data)
    plot_confusion_matrix(all_test_label,prediction,output_dir,title=f"{TEST_PATIENT}-TW-Augmented-Prediciton")
    aug_acc, aug_f1 = accuracy_score(all_test_label,prediction), f1_score(all_test_label,prediction,average="micro")
    logger.info(f"{TEST_PATIENT}(WITH Time-Warping AUGMENTATION): Accuracy: {aug_acc*100:.2f}% | F1-score: {aug_f1*100:.2f}%")

    # train with diffusion augmentation
    logger.info("Train with Diffusion Augmentation")
    train_dataset = ULF_Classification_Dataset(np.concatenate([all_train_data,all_train_data_aug_diffusion],axis=0),np.concatenate([all_train_label,all_train_label_aug_diffusion],axis=0))
    train_dataloader = DataLoader(train_dataset,cc_config['batch_size'],shuffle=True)
    trainer.load_weight(os.path.join(ckpt_dir,"initial.pth"))
    trainer.train(train_dataloader,test_dataloader,args.max_ci,os.path.join(ckpt_dir,f'{TEST_PATIENT}_best_aug.pth'),verbal=args.verbal,writer=writer)
    trainer.load_weight(os.path.join(ckpt_dir,f'{TEST_PATIENT}_best_aug.pth'))
    prediction = trainer.make_prediction(all_test_data)
    plot_confusion_matrix(all_test_label,prediction,output_dir,title=f"{TEST_PATIENT}-Diffusion-Augmented-Prediciton")

    diffusion_aug_acc, diffusion_aug_f1 = accuracy_score(all_test_label,prediction), f1_score(all_test_label,prediction,average="micro")
    logger.info(f"{TEST_PATIENT}(WITH Diffusion AUGMENTATION): Accuracy: {diffusion_aug_acc*100:.2f}% | F1-score: {diffusion_aug_f1*100:.2f}%")
    
    return orig_acc, orig_f1, aug_acc, aug_f1


total_orig_acc, total_orig_f1, total_aug_acc, total_aug_f1 = [] , [] , [] , []
for patient in args.test_patient:
    orig_acc, orig_f1, aug_acc, aug_f1 = main(patient)
    total_orig_acc.append(orig_acc)
    total_orig_f1.append(orig_f1)
    total_aug_acc.append(aug_acc)
    total_aug_f1.append(aug_f1)


logger.info(f"Total Original Accuracy:{(sum(total_orig_acc)/len(total_orig_acc))*100:.4f}%")
logger.info(f"Total Original F1-score:{(sum(total_orig_f1)/len(total_orig_f1))*100:.4f}%")
logger.info(f"Total Augmented Accuracy:{(sum(total_aug_acc)/len(total_aug_acc))*100:.4f}%")
logger.info(f"Total Augmented Accuracy:{(sum(total_aug_f1)/len(total_aug_f1))*100:.4f}%")


