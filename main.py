import tsgm
from argument import parse_argument

from dataset import Dataset


if __name__ == '__main__':
    args = parse_argument()

    dataset = Dataset(args.data,args.config)
    (train_data, train_label), (test_data, test_label) = dataset.get_dataset(test_patient=['P29','P30'])

    print(train_data.shape,test_data.shape)