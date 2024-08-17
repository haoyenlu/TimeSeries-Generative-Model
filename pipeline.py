import numpy as np
import argparse
import os
from collections import defaultdict
from datetime import datetime

from utils import load_config
from data_utils import FeatureWiseScaler , WindowWarping
from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample


curr_date = datetime.now().strftime("%d%m%Y_%H%M%S")

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--test_patient','-tp',nargs='+',type=str)
parser.add_argument('--include_healthy','-ih',action='store_true')
parser.add_argument('--classification_config','-cc',type=str)
parser.add_argument('--output','-o',type=str)
args = parser.parse_args()


# DIRECTORY
output_dir = os.path.join(args.output,curr_date)
os.makedirs(output_dir)

# TODO: preprocess dataset
# Data Structure - Healthy - H01 ~ H05 - TASK- data
#                - Stroke - P02 ~ P30 - TASK - data

data = np.load(args.data,allow_pickle=True).item()
train_dataset = defaultdict(list)
test_dataset = defaultdict(list)

for type, type_dict in data.items():
    if not args.include_healthy and type == 'Healthies': continue

    for patient , patient_dict in type_dict.items():
        for task , task_data in patient_dict.items():
            if patient not in args.test_patient:
                train_dataset[task].append(task_data)
            else:
                test_dataset[task].append(task_data)


# TODO: Data augmentation and Preprocessing
scaler = FeatureWiseScaler(feature_range=(0,1))
augmenter = WindowWarping(window_ratio=0.2,scales=[0.1,0.5,1,1.5,2,2.5])
tasks = train_dataset.keys()

all_data = []
all_data_aug = []
all_label = []
all_label_aug = []
for task in tasks:
    train_data = np.concatenate(train_dataset[task],axis=0)
    train_data = scaler.fit_transform(train_data)
    train_data_aug = augmenter.generate(train_data)

    # plot_sample(train_data,train_data_aug,output_dir)
    # plot_pca(train_data,train_data_aug,output_dir)
    # plot_tsne(train_data,train_data_aug,output_dir)
    # plot_umap(train_data,train_data_aug,output_dir)
    

    # TODO: train generative model on train_data

    # TODO: generate dataset with label
    label = np.argwhere(tasks == task)
    all_data.append(train_data)
    all_label.append([label] * train_data.shape[0])
    all_data_aug.append(train_data_aug)
    all_label_aug.append([label] * train_data_aug.shape[0])

all_data = np.concatenate(all_data,axis=0)
all_label = np.concatenate(all_label,axis=0)
all_data_aug = np.concatenate(all_data_aug,axis=0)
all_label_aug = np.concatenate(all_label_aug,axis=0)


print(all_data.shape,all_label.shape,all_data_aug.shape,all_label_aug.shape)

# TODO: train classification on augmentation and original dataset and test with test dataset




    