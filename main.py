import tsgm
from argument import parse_argument
import numpy as np
import tensorflow as tf
import yaml

from dataset import PreprocessMVNX
from utils import save_to_numpy



def preprocess_mvnx(args,config):
    preprocessor = PreprocessMVNX(config=config)
    (train_data, train_label), (test_data, test_label) = preprocessor.get_dataset(path=args.data,test_patient=args.test_patient)


    if args.save is not None:
        save_path = f'./{args.save}/ulf_original_{"_".join(args.test_patient)}.npy'
        save_to_numpy(train_data,train_label,test_data,test_label,path=save_path)

    print(train_data.shape,test_data.shape)

    # visualize_ts_lineplot(train_data,train_label,path='./visualize/original.png')

    scaler = tsgm.utils.TSFeatureWiseScaler((-1,1))
    encoder = OneHotEncoder(handle_unknown='ignore')
    scaler.fit(np.concatenate([train_data,test_data],axis=0))
    encoder.fit(np.array(config['tasks']).reshape(-1,1))
    X_train = scaler.transform(train_data)
    Y_train = encoder.transform(train_label).toarray()

    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)

    # visualize_ts_lineplot(X_train,train_label,path='./visualize/scale.png')

    X_test = scaler.transform(test_data)
    Y_test = encoder.transform(test_label).toarray()

    print(Y_train[0])

    if args.save is not None:
        save_path = f'./{args.save}/ulf_rescale_{"_".join(args.test_patient)}.npy'
        save_to_numpy(X_train,Y_train,X_test,Y_test,path=save_path)


def to_tensor_dataset(X,Y,buffer_size,batch_size):
    dataset = tf.data.Dataset.from_tensor_slice((X,Y))
    dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)



if __name__ =='__main__':
    args = parse_argument()
        
    with open(args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)

    preprocess_mvnx(args,config)