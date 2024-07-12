import numpy as np
import tensorflow as tf
import yaml

def save_to_numpy(train_data,train_label,test_data,test_label,path):
    save = {'train':{'data':train_data,'label':train_label},'test':{'data':test_data,'label':test_label}}
    np.save(path,save)

def load_numpy_data(path):
    data = np.load(path,allow_pickle=True).item()
    train_data = data['train']['data']
    train_label = data['train']['label']
    test_data = data['test']['data']
    test_label = data['test']['label']

    return (train_data, train_label), (test_data, test_label)


def to_tensor_dataset(X,y,buffer_size=1024,batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset

def load_config(path):
    with open(path.config) as file:
        config = yaml.safe_load(file)

    return config

