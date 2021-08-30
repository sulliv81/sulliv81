#! /usr/bin/python3

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)

An example of a linear model in PyTorch Lightning for CIFAR10 classification.

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.

"""
import argparse

import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import os
import torch.nn as nn
args = None

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("-opt", type=str, \
                        help="The optimizer: \"adadelta\",  \"adagrad\", \"rmsprop\", \"sgd\" "
                             " (string) [default: \"adam\"]", default="adam")
    parser.add_argument("-lr", type=float, \
                        help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-mb", type=int, \
                        help="The minibatch size (int) [default: 32]", default=32)
    parser.add_argument("-epochs", type=int, \
                        help="The number of training epochs (int) [default: 100]", \
                        default=100)

    return parser.parse_args()

args = parse_all_args()

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self,pin_memory=False,num_workers=0,val_n=5000):
        super().__init__()

        self.mb = args.mb
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_n = val_n

        self.transform=transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # download data if not downloaded
        CIFAR10(os.getcwd(),train=True,download=True)
        CIFAR10(os.getcwd(),train=False,download=True)

    def setup(self,step=None):
        # make splits, create Dataset objects

        if step == 'fit' or step is None:
            # Load train and val
            trainvalset = CIFAR10(os.getcwd(),train=True,
                    transform=self.transform)
            self.trainset,self.valset = random_split(trainvalset,
                    [len(trainvalset)-self.val_n,self.val_n])

        if step == 'test' or step is None:
            # Load test set
            self.testset = CIFAR10(os.getcwd(),train=False,
                    transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.mb,
                pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.mb,
                pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.mb,
                pin_memory=self.pin_memory, num_workers=self.num_workers)



class MultiLogReg(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.lr = args.lr

        self.accuracy = Accuracy()

    def forward(self, x):
        #batch_size,_,_,_ = x.size()
        #x = x.view(batch_size, -1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def eval_batch(self, batch, batch_idx):
        # Make predictions
        x, y = batch
        y_pred = self(x)

        # Evaluate predictions
        loss = F.cross_entropy(y_pred, y)
        acc = self.accuracy(y_pred, y)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)

        x,y = batch
        y_pred = self(x)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)


        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)

        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optVal = args.opt
        if (optVal == "adam"):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif (optVal == "adagrad"):
            return torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif (optVal == "rmsprop"):
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif (optVal == "sgd"):
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        elif (optVal == "adadelta"):
            return torch.optim.Adadelta(self.parameters(), lr=self.lr)
        return

def setOpt(modelParams, lrVal, optVal):

    #check if -lr given
    if (optVal == "adam"):
        return torch.optim.Adam(modelParams, lr=lrVal)
    elif (optVal == "adagrad"):
        return torch.optim.Adagrad(modelParams, lr=lrVal)
    elif (optVal == "rmsprop"):
        return torch.optim.RMSprop(modelParams, lr=lrVal)
    elif (optVal == "sgd"):
        return torch.optim.SGD(modelParams, lr=lrVal)
    elif (optVal ==  "adadelta"):
        return torch.optim.Adadelta(modelParams, lr=lrVal)

def main():
    args = parse_all_args()
    data  = CIFAR10DataModule()
    model = MultiLogReg()

    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model,data)
    trainer.test(model,datamodule=data)

if __name__ == '__main__':
    main()
