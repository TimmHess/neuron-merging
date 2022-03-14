from __future__ import print_function

import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import pickle
import copy

cwd = os.getcwd()
sys.path.append(cwd+'/../')
import models
from torchvision import datasets, transforms
from torch.autograd import Variable
from lib.decompose import Decompose
from lib.neuron_merger import NeuronMerger

from cmd_parser import parser

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            print(key)
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
                    
    # save
    if args.model_type == 'original':
        if args.arch == 'WideResNet' :
            model_filename = '.'.join([args.arch,
                                    args.dataset,
                                    args.model_type,
                                    '_'.join(map(str, args.depth_wide)),
                                    'pth.tar'])
        elif args.arch == 'ResNet' :
            model_filename = '.'.join([args.arch,
                                    args.dataset,
                                    args.model_type,
                                    str(args.depth_wide),
                                    'pth.tar'])
        else:
            model_filename = '.'.join([args.arch,
                                        args.dataset,
                                        args.model_type,
                                        'pth.tar'])
    else: # retrain
        if args.arch == 'WideResNet' :
            model_filename = '.'.join([args.arch,
                                    '_'.join(map(str, args.depth_wide)),
                                    args.dataset,
                                    args.model_type,
                                    args.criterion,
                                    str(args.pruning_ratio),
                                    'pth.tar'])
        elif args.arch == 'ResNet' :
            model_filename = '.'.join([args.arch,
                                    str(args.depth_wide),
                                    args.dataset,
                                    args.model_type,
                                    args.criterion,
                                    str(args.pruning_ratio),
                                    'pth.tar'])
        else :
            model_filename = '.'.join([args.arch,
                                    args.dataset,
                                    args.model_type,
                                    args.criterion,
                                    str(args.pruning_ratio),
                                    'pth.tar'])

    torch.save(state, os.path.join('saved_models/', model_filename))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    return


def test(epoch, evaluate=False):
    global best_acc
    global best_epoch
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * float(correct) / len(test_loader.dataset)

    if (acc > best_acc):
        best_acc = acc
        best_epoch = epoch
        if not evaluate:
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%, Best Epoch: {}\n'.format(best_acc, best_epoch))
    return


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    lr = args.lr
    for (gamma, step) in zip (gammas, schedule):
        if(epoch>= step) and (args.epochs * 3 //4 >= epoch):
            lr = lr * gamma
        elif(epoch>= step) and (args.epochs * 3 //4 < epoch):
            lr = lr * gamma * gamma
        else:
            break
    print('learning rate : ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return


if __name__=='__main__':
    args = parser.parse_args()
    # check options
    if not (args.model_type in [None, 'original', 'merge', 'prune']):
        print('ERROR: Please choose the correct model type')
        exit()
    if not (args.target in [None, 'conv', 'ip']):
        print('ERROR: Please choose the correct decompose target')
        exit()
    if not (args.arch in ['VGG','ResNet','WideResNet','LeNet_300_100', 'SimpleCNN']):
        print('ERROR: specified arch is not suppported')
        exit()
    
    if(not(args.seed == -1)):
        torch.manual_seed(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        if(not(args.seed == -1)):
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic=True

    # load data
    num_classes = args.num_classes

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        
        num_classes = 10

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        
        num_classes = 100

    elif args.dataset == 'FashionMNIST':
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        num_classes = 10

    elif args.dataset == 'MNIST':
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
        train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        num_classes = 10

    else : 
        pass


    if args.depth_wide:
            args.depth_wide = eval(args.depth_wide)

    cfg = None

    # make cfg
    if args.retrain:
        if args.target == 'conv' :
            if args.arch == 'SimpleCNN':
                cfg = [8, "M", 16, "M", 32, "M"]
                for i in range(len(cfg)):
                    if(type(cfg[i]) == int) and i > 2:
                        cfg[i] = int(cfg[i] * (1 - args.pruning_ratio)) 
                temp_cfg = list(filter(('M').__ne__, cfg))
                print("SimpleCNN cfg: ", temp_cfg)

            if args.arch == 'VGG':
                if args.dataset == 'cifar10':
                    #cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]
                    cfg = [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256]
                elif args.dataset == 'cifar100':
                    cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 256, 'M', 256, 256, 256]

                for i in range(len(cfg)):
                    if(type(cfg[i])==int and i > 3):
                       cfg[i] = int(cfg[i] * (1 - args.pruning_ratio)) 
                temp_cfg = list(filter(('M').__ne__, cfg))
                print("VGG cfg: ", temp_cfg)

            elif args.arch == 'ResNet':
                cfg = [16, 32, 64]
                for i in range(len(cfg)):
                    cfg[i] = int(cfg[i] * (1 - args.pruning_ratio))
                temp_cfg = cfg

            elif args.arch == 'WideResNet':
                cfg = [16, 32, 64]
                temp_cfg = [16, 32, 32]

                for i in range(len(cfg)):
                    cfg[i] = int(cfg[i] * (1 - args.pruning_ratio))
                    temp_cfg[i] = cfg[i] * args.depth_wide[1]
        
        elif args.target == 'ip' :
            if args.arch == 'LeNet_300_100':
                cfg = [300,100]
                for i in range(len(cfg)):
                    cfg[i] = round(cfg[i] * (1 - args.pruning_ratio))
                temp_cfg = cfg
            pass


    # generate the model
    model = models.generate_model(args.arch, cfg, num_classes, args)
    print(model)
    if args.cuda:
        model.cuda()

    # pretrain
    best_acc = 0.0
    best_epoch = 0
    if args.pretrained:
        pretrained_model = torch.load(args.pretrained)
        best_epoch = 0
        if args.model_type == 'original':
            best_acc = pretrained_model['acc']
            model.load_state_dict(pretrained_model['state_dict'])



    # weight initialization
    if args.retrain:
        decomposed_list = Decompose(args.arch, pretrained_model['state_dict'], args.criterion, args.threshold, 
                            args.lamda, args.model_type, temp_cfg, args.cuda, args.no_bn).main()
        model = models.weight_init(model, decomposed_list)


    # print the number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total parameter number:', params, '\n')


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()


    if args.evaluate:
        test(0, evaluate=True)
        exit()

    """
    # DEBUG
    if(args.arch == "VGG"):
        neuronMerger = NeuronMerger(expand_percentage=0.2, args=args)
        print("NeuronMerger Initialized...")
        print(model)
        model = neuronMerger.expand(model)
        print("Model got expanded...")
        #print(model)
        #sys.exit()
        model = neuronMerger.compress(model)
        print("Model got compressed...")
        #print(model)
    print("done...")
    sys.exit()
    # END DEBUG
    """
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        train(epoch)
        test(epoch)