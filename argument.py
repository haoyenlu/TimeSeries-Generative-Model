import argparse


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str)
    parser.add_argument('--config',type=str)
    parser.add_argument('--save',type=str)
    parser.add_argument('--test_patient',nargs='*')



    args = parser.parse_args()
    return args

