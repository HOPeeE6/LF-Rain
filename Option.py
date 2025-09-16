import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--A', type=int, default=5)
parser.add_argument('--log_path', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_min', type=float, default=1e-6)
parser.add_argument('--train_path', type=str, default='')
parser.add_argument('--val_path', type=str, default='', )
parser.add_argument('--num_heads', type=int, default=, )
parser.add_argument('--epoch', type=int, default=)
parser.add_argument('--channel', type=int, default=)
parser.add_argument('--val_after_every', type=int, default=)
parser.add_argument('--win_size', type=int, default=)
parser.add_argument('--stride', type=int, default=[, ])
parser.add_argument('--preprocess', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--Resume', type=bool, default=False)


args = parser.parse_args()
