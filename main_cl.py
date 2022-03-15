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
from tqdm import tqdm


cwd = os.getcwd()
sys.path.append(cwd+'/../')
import models

"""
counter = 0
for i, exp in enumerate(train_stream):
    dataset, t = exp.dataset, exp.task_label
    
    dataloader = DataLoader(dataset, batch_size=32)

    for batch in dataloader:
        x, y, *other = batch
        print(x.shape)
        print(y.shape)
    
        img = cv2.cvtColor(np.asarray(ToPILImage()(x[0]), dtype=np.uint8), cv2.COLOR_RBG2BGR)
        print(img.shape)
        cv2.imwrite("img" + str(counter) + ".png", img)
        counter += 1
        break
"""


def train_one_epoch(epoch, train_loader):
    for batch_idx, batch in enumerate(train_loader):
        data, target, _ = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
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

def test(epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader)):
            data, target, _ = batch
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            test_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        acc = 100. * float(correct) / len(test_loader.dataset)

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss * args.batch_size, correct, len(test_loader.dataset),
            100. * float(correct) / len(test_loader.dataset)))


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

# Create Model
cfg = [16, "M", 32, "M", 64, "M", 64, "M"]
model = models.generate_model(args.arch, num_classes=args.num_classes, args=args, cfg=cfg)
print(model)
if args.cuda:
    model.cuda()

# Load Pretrained Weights
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

# Create Dataset - Create CL Benchmark
transform = transforms.Compose([transforms.ToTensor()])
scenario = EndlessCLSim(
    scenario="Illumination",
    sequence_order=args.seq_order,
    task_order=None,
    dataset_root="./data/",
    train_transform=transform,
    eval_transform=transform
)
train_stream = scenario.train_stream
test_stream = scenario.test_stream
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print("Train stream:", len(train_stream))
print("Val stream:", len(test_stream))

if args.evaluate and not args.retrain:
    print("Evaluation only...")
    # Test Loop
    for j, test_exp in enumerate(test_stream):
        test_set, t = test_exp.dataset, test_exp.task_label
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        test(0, test_loader)
    sys.exit()

for i, exp in enumerate(train_stream):
    dataset, t = exp.dataset, exp.task_label
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, **kwargs)
    
    # Train Loop
    for epoch in range (args.epochs):
        train_one_epoch(epoch, train_loader)

    # Store model
    if not args.retrain:
        models.save_state(model, -1.0, args)

    # Test Loop
    for j, test_exp in enumerate(test_stream):
        test_set, t = test_exp.dataset, test_exp.task_label
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        test(0, test_loader)


