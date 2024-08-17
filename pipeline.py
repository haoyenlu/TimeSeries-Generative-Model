import numpy as np
import argparse

from collections import defaultdict
from datetime import datetime

from data_utils import FeatureWiseScaler , DTWBarycentricAveraging
from analysis_utils import plot_pca, plot_tsne, plot_umap, plot_sample

import logging

logger = logging.getLogger('main')
logging.basicConfig()

curr_date = datetime.now().strftime("%d%m%Y_%H%M%S")

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str)
parser.add_argument('--test_patient','-tp',nargs='+',type=str)
parser.add_argument('--include_healthy','-ih',action='store_true')
parser.add_argument('--save','-s',type=str)
args = parser.parse_args()


# TODO: preprocess dataset
# Data Structure - Healthy - H01 ~ H05 - TASK- data
#                - Stroke - P02 ~ P30 - TASK - data


logger.info('Loading Data')
data = np.load(args.data,allow_pickle=True).item()


train_dataset = defaultdict(list)
test_dataset = defaultdict(list)

logger.info('Processing Data')
for type, type_dict in data.items():
    if not args.include_healthy and type == 'Healthies': continue

    for patient , patient_dict in type_dict.items():
        for task , task_data in patient_dict.items():
            if patient not in args.test_patient:
                train_dataset[task].append(task_data)
            else:
                test_dataset[task].append(task_data)


logger.info('Preprocessing and Augmenting Data')
# TODO: Data augmentation and Preprocessing
scaler = FeatureWiseScaler(feature_range=(0,1))
DTW = DTWBarycentricAveraging()
tasks = train_dataset.keys()

for task in tasks:
    train_data = np.concatenate(train_dataset[task],axis=0)
    train_data = scaler.fit_transform(train_data)
    train_data_aug = DTW.generate(train_data, n_samples=100)

    plot_pca(real=train_data,fake=train_data_aug,save_path=args.save)
    plot_tsne(real=train_data,fake=train_data_aug,save_path=args.save)
    plot_umap(real=train_data,fake=train_data_aug,save_path=args.save)
    plot_sample(real=train_data,fake=train_data_aug,save_path=args.save)
    break



# TODO: train generative model on train_dataset



    