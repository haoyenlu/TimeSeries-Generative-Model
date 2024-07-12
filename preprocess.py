import argparse
import yaml
from sklearn.preprocessing import OneHotEncoder
import tsgm
import numpy as np

from dataset import PreprocessMVNX
from utils import save_to_numpy, load_numpy_data

parser = argparse.ArgumentParser()
parser.add_argument('--mvnx',type=str,default=None)
parser.add_argument('--numpy',type=str,default=None)
parser.add_argument('--config',type=str)
parser.add_argument('--test_patient',nargs='*')

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
scale_range = (-1,1)
print(f"Scaling time-series sequence to {scale_range}.")
scaler = tsgm.utils.TSFeatureWiseScaler(scale_range)
scaler.fit(np.concatenate([train_data,test_data],axis=0))
X_train = scaler.transform(train_data).astype(np.float32)
X_test = scaler.transform(test_data).toarray().astype(np.float32)

'''Onehot Encoding'''
print("Encoding label.")
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(np.array(config['tasks']).reshape(-1,1))
Y_train = encoder.transform(train_label).astype(np.float32)
Y_test = encoder.transform(test_label).toarray().astype(np.float32)

save_path = f'./{args.save}/ulf_preprocess_{"_".join(args.test_patient)}.npy'
save_to_numpy(X_train,Y_train,X_test,Y_test,save_path)




