import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gpu',
    default = '0',
    type = str,
    help='choose gpu device')
parser.add_argument(
    '--dataset',
    default = 'YAGO-WIKI180K',
    type = str,
    help='choose dataset')
parser.add_argument(
    '--seed',
    default = 2000,
    type = int,
    help='choose the number of align seeds')
parser.add_argument(
    '--dropout',
    default = 0.3,
    type = float,
    help='choose dropout rate')
parser.add_argument(
    '--depth',
    default = 2,
    type = int,
    help='choose number of GNN layers')
parser.add_argument(
    '--gamma',
    default = 1.0,
    type = float,
    help='choose margin')
parser.add_argument(
    '--lr',
    default = 0.005,
    type = float,
    help='choose learning rate')
parser.add_argument(
    '--dim',
    default = 100,
    type = int,
    help='choose embedding dimension')
parser.add_argument(
    '--randomseed',
    default=42,
    type=int,
    help='choose random seed'
)
parser.add_argument(
    '--batchsize',
    default=512,
    type=int,
    help='choose batch size'
)
parser.add_argument(
    '--nthread',
    default = 30,
    type = int,
    help='choose number of threads for computing time similarity ')
parser.add_argument(
    '--iteration',
    default = 3,
    type = int,
    help='choose the iteration number'
)

args = parser.parse_args()