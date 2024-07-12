import numpy as np



def load_numpy_data(path):
    data = np.load(path,allow_pickle=True).item()
    train_data = data['train']['data']
    train_label = data['train']['label']
    test_data = data['test']['data']
    test_label = data['test']['label']

    return (train_data, train_label), (test_data, test_label)