import yaml
import numpy as np
import os
from tqdm import tqdm
from mvnx import PreprocessMVNX

from argument import preprocess_argument , preprocess_synthesize_argument




def preprocess_original():
    args = preprocess_argument()

    with open(args.config) as file:
        config = yaml.safe_load(file)


    assert args.mvnx is not None

    os.makedirs(args.save,exist_ok=True)

    preprocessor = PreprocessMVNX(**config)
    data = preprocessor.get_dataset(args.mvnx)   

    # Train data
    save_path = f'./{args.save}/ulf_new_{"_".join(args.test_patient)}.npy'
    np.save(save_path,data)




def preprocess_synthesize():
    args = preprocess_synthesize_argument()

    synthesize_data = {}

    for task in tqdm(os.listdir(args.data)):
        file = os.path.join(args.data,task,"synthesize.npy")
        
        synthesized = np.load(file,allow_pickle=True)
        synthesize_data[task] = synthesized

    
    np.save(os.path.join(args.save,'ulf_all_task_synthesize.npy'),synthesize_data)

        







if __name__ == '__main__':
    preprocess_original()