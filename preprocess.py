import argparse
import yaml
import numpy as np
import os

from mvnx import PreprocessMVNX
from utils import save_to_numpy, load_numpy_data
from data_utils import FeatureWiseScaler

from argument import preprocess_argument




def main():
    args = preprocess_argument()

    with open(args.config) as file:
        config = yaml.safe_load(file)


    assert args.mvnx is not None

    os.makedirs(args.save,exist_ok=True)

    preprocessor = PreprocessMVNX(**config)
    task_data, test_task_data = preprocessor.get_dataset(args.mvnx,args.test_patient)   

    # Train data
    save_path = f'./{args.save}/ulf_all_task.npy'
    np.save(task_data,save_path)

    # Test Data
    save_path = f'./{args.save}/ulf_all_task{"_".join(args.test_patient)}.npy'
    np.save(test_task_data,save_path)






if __name__ == '__main__':
    main()