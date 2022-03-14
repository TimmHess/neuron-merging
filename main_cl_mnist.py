import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor
from avalanche.benchmarks.classic import EndlessCLSim

from cmd_parser import parser
from lib.neuron_merger import NeuronMerger

import numpy as np
import cv2
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd+'/../')
import models

# Get Args
args = parser.parse_args()

if(not(args.seed == -1)):
    torch.manual_seed(args.seed)
# Get Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    if(not(args.seed == -1)):
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True

# Create Dataset
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

num_classes = 10

# Create Model
model = models.generate_model(args.arch, num_classes=num_classes, args=args, cfg=None)
print(model)
if args.cuda:
    model.cuda()

# pretrain
if args.pretrained:
    pretrained_model = torch.load(args.pretrained)
    model.load_state_dict(pretrained_model['state_dict'])
    print("Loaded Pretrained model...")

# Create Neuron Merger
neuronMerger = NeuronMerger(expand_percentage=args.pruning_ratio, args=args)
if(args.retrain):
    neuronMerger.store_weight_shapes(model)
    model = neuronMerger.expand(model)
    print("expanded model")
    print(model)

# Create Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

# Train Loop
model.train()
for epoch in range (args.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # NeuronMerger freeze
        #neuronMerger.freeze_old_weights(model)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


# Test Loop
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * float(correct) / len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))


# Compress model back to original size
model = neuronMerger.compress(model)
#model = neuronMerger.compress_deterministic(model)
print(model)

# Test Loop
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * float(correct) / len(test_loader.dataset)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
