import argparse
from curses import meta

parser = argparse.ArgumentParser(description='Neuron Merging Argument Parser')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
        help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
        help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
        help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
        metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
        help='random seed (default: -1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status')
parser.add_argument('--arch', action='store', default='VGG',
        help='network structure: VGG | ResNet | WideResNet | LeNet_300_100')
parser.add_argument('--num_classes', type=int, default=10, metavar='NC',
        help='number of final classification classes')
parser.add_argument('--pretrained', action='store', default=None,
        help='pretrained model')
parser.add_argument('--evaluate', action='store_true', default=False,
        help='whether to run evaluation')
parser.add_argument('--retrain', action='store_true', default=False,
        help='whether to retrain')
parser.add_argument('--model-type', action='store', default='original',
        help='model type: original | prune | merge')
parser.add_argument('--target', action='store', default='conv',
        help='decomposing target: default=None | conv | ip')
parser.add_argument('--dataset', action='store', default='cifar10',
        help='dataset: cifar10 | cifar100 | FashionMNIST')
parser.add_argument('--criterion', action='store', default='l1-norm',
        help='criterion : l1-norm | l2-norm | l2-GM')
parser.add_argument('--threshold', type=float, default=1,
        help='threshold (default: 1)')
parser.add_argument('--lamda', type=float, default=0.8,
        help='lamda (default: 0.8)')
parser.add_argument('--pruning-ratio', type=float, default=0.7,
        help='pruning ratio : (default: 0.7)')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1,0.1],
        help='gammas : (default: [0.1,0.1])')
parser.add_argument('--schedule', type=int, nargs='+', default=[100,200],
        help='schedule : (default: [100,200])')
parser.add_argument('--depth-wide', action='store', default=None,
        help='depth and wide (default: None)')
parser.add_argument('--no_bn', action='store_true', default=False,
        help='drops batch-norm term in neuron merging function') # TODO: make this agnistic to whether ther actually is a batch-norm term


parser.add_argument('--seq_order', type=int, nargs='+', default=[0,1,2,3,4],
        help='indices of subsequent sequences')