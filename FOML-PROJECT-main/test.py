import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from random import shuffle
import torch
from torch import nn
import math
from glob import glob
import sys
import shutil  


import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL
import random
from scipy import ndimage
from dataset import segDataset,to_device,get_default_device
from loss import FocalLoss
from Unet import UNet
from utils import acc
from Unet import UNet

color_shift = transforms.ColorJitter(.1,.1,.1,.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

t = transforms.Compose([color_shift, blurriness])
dataset = segDataset(r"E:\deployment_spam\UNet\Semanticsegmentationdataset" , training = True, transform= t)

test_num = int(0.1 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_num, test_num], generator=torch.Generator().manual_seed(101))

BACH_SIZE = 4
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=0)

class DeviceDataLoader():
    # move the batches of the data to our selected device
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

device = get_default_device()

train_dataloader = DeviceDataLoader(train_dataloader, device)
test_dataloader = DeviceDataLoader(test_dataloader, device)

criterion = FocalLoss(gamma=3/4).to(device)
min_loss = torch.tensor(float('inf'))

model =  UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
model.load_state_dict(torch.load(r"E:\deployment_spam\unet_epoch_0_0.89130.pt"))

model.eval()
for batch_i, (x, y) in enumerate(test_dataloader):
    for j in range(len(x)):
        result = model(x[j:j+1])
        mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
        im = np.moveaxis(x[j].cpu().detach().numpy(), 0, -1).copy()*255
        im = im.astype(int)
        gt_mask = y[j].cpu()

        plt.figure(figsize=(12,12))

        plt.subplot(1,3,1)
        im = np.moveaxis(x[j].cpu().detach().numpy(), 0, -1).copy()*255
        im = im.astype(int)
        plt.imshow(im)

        plt.subplot(1,3,2)
        plt.title("orginal maks")
        plt.imshow(gt_mask)

        plt.subplot(1,3,3)
        plt.title("predicted mask")
        plt.imshow(mask)
        plt.show()
