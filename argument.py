import argparse



def train_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str)
    parser.add_argument('--config',type=str)
    parser.add_argument('--max_iter',type=int,default=1000)
    parser.add_argument('--save_iter',type=int,default=100)
    parser.add_argument('--n_critic',type=int,default=5) # Only for GAN
    parser.add_argument('--log',type=str,default=None)
    parser.add_argument('--ckpt',type=str,default='./checkpoint')
    parser.add_argument('--load_ckpt',type=str,default=None)
    parser.add_argument('--task',nargs='+',type=str)
    parser.add_argument('--save',type=str)
    parser.add_argument('--verbal',action='store_true')
    

    args = parser.parse_args()
    return args


def train_classification_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--config',type=str)
    parser.add_argument('--max_iter',type=int)
    parser.add_argument('--save_iter',type=int)
    parser.add_argument('--log',type=str)
    parser.add_argument('--ckpt',type=str)
    parser.add_argument('--save',type=str)

    args = parser.parse_args()
    return args


def preprocess_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mvnx',type=str,default=None)
    # parser.add_argument('--numpy',type=str,default=None)
    parser.add_argument('--config',type=str)
    parser.add_argument('--test_patient',nargs='*')
    parser.add_argument('--save',type=str)

    args = parser.parse_args()
    return args

def preprocess_synthesize_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str)
    parser.add_argument('--save',type=str)

    args = parser.parse_args()
    return args


def analysis_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str)
    parser.add_argument('--ckpt',type=str)
    parser.add_argument('--config',type=str)
    parser.add_argument('--save',type=str)
    parser.add_argument('--task',type=str,default='T01')
    parser.add_argument('--curr_date',type=str)

    args = parser.parse_args()
    return args