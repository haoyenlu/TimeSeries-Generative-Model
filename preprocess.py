import argparse
import yaml
import numpy as np

from mvnx import PreprocessMVNX
from utils import save_to_numpy, load_numpy_data
from data_utils import FeatureWiseScaler

parser = argparse.ArgumentParser()
parser.add_argument('--mvnx',type=str,default=None)
parser.add_argument('--numpy',type=str,default=None)
parser.add_argument('--config',type=str)
parser.add_argument('--test_patient',nargs='*')
parser.add_argument('--save',type=str)

args = parser.parse_args()


with open(args.config) as file:
    config = yaml.safe_load(file)


if args.mvnx is not None:
    preprocessor = PreprocessMVNX(config)
    (train_data,train_label) , (test_data,test_label) = preprocessor.get_dataset(args.mvnx,args.test_patient)

    save_path = f'./{args.save}/ulf_original_{"_".join(args.test_patient)}.npy'
    save_to_numpy(train_data,train_label,test_data,test_label,save_path)

elif args.numpy is not None:
    (train_data,train_label) , (test_data,test_label) = load_numpy_data(args.numpy)

else:
    raise Exception("Specify either mvnx or numpy file to load.")


'''Scaling'''
scale_range = config['scale_range']
scaler = FeatureWiseScaler(scale_range)
scaler.fit(np.concatenate([train_data,test_data],axis=0))
X_train = scaler.transform(train_data).astype(np.float32)
X_test = scaler.transform(test_data).astype(np.float32)



save_path = f'./{args.save}/ulf_preprocess_{"_".join(args.test_patient)}.npy'
save_to_numpy(X_train,train_label,X_test,test_label,save_path)




