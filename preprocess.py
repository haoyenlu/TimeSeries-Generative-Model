import yaml
import numpy as np
import os
from tqdm import tqdm
from mvnx import PreprocessMVNX

from argument import preprocess_argument , preprocess_synthesize_argument




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
    np.save(save_path,task_data)

    # Test Data
    save_path = f'./{args.save}/ulf_all_task{"_".join(args.test_patient)}.npy'
    np.save(save_path,test_task_data)



def preprocess_synthesize():
    args = preprocess_synthesize_argument()

    synthesize_data = {}

    for task in tqdm(os.listdir(args.data)):
        curr_path = os.path.join(args.data,task)
        
        synthesized = np.load(curr_path,"synthesize.npy",allow_pickle=True)
        synthesize_data[task] = synthesized

    
    np.save(os.path.join(args.save,'ulf_all_task_synthesize.npy'),synthesize_data)

        







if __name__ == '__main__':
    main()