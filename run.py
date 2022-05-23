from train import Train
import argparse
from config import Config

def str2bool(s):
    return s.lower() in ("yes", "true", "t", "1")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, required=False, help="ID of GPU")
    parser.add_argument("--epochs", type=int, required=False, help="training epochs")
    parser.add_argument('--train_seg', type=str, required=False, help="train part")
    parser.add_argument('--train_dec', type=str, required=False, help="train part")
    parser.add_argument('--train_total', type=str, required=False, help="train part")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    config = Config()
    config.merge_from_args(args)
    config.init_extra()

    train = Train(cfg=config)
    train.train()