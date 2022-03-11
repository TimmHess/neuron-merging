from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from avalanche.benchmarks.classic import EndlessCLSim

from cmd_parser import parser

import numpy as np
import cv2

import sys


# Create CL Benchmark
scenario = EndlessCLSim(
    scenario="Illumination",
    sequence_order=None,
    task_order=None,
    dataset_root="./data/"
)

train_stream = scenario.train_stream
test_stream = scenario.test_stream


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