'''
Bo Sullivan and Max Lisaius

Task 1 Final Project

'''
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio
import argparse

torchaudio.set_audio_backend = "sox_io"

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


class NotADataset(Dataset):
    #Dataset to index the list and be used with our data loader
    def __init__(self, datalist, seq_len: int = 20):
        self.Xlist = datalist
        self.length = len(datalist)
        #self.seq_len = seq_len
        #self.num_speakers = len(self.Xlist)
    #returns total indicies
    def __len__(self):
        return self.length
    #returns specific index
    def __getitem__(self, index):

        return (Xlist[index] , 1)



class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self,pin_memory=False,num_workers=4,val_n=5000):
        super().__init__()

        self.mb = args.mb
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_n = val_n

        self.transform=transforms.Compose([transforms.ToTensor()])

    #def prepare_data(self):
        # download data if not downloaded
        #CIFAR10(os.getcwd(),train=True,download=True)
        #CIFAR10(os.getcwd(),train=False,download=True)

    def setup(self,step=None):
        # make splits, create Dataset objects

        if step == 'fit' or step is None:


            files = np.loadtxt('task1traindata.txt', dtype='int')
            print(files)
            datalist = []
            for file in files:
                filename = "notdata/file" + str(file) +".wav"
                waveform, sample_rate = torchaudio.load(filename)
                print("Shape of waveform: {}".format(waveform.size()))
                print("Sample rate of waveform: {}".format(sample_rate))

                wav = waveform.flatten()
                print(wav.shape)
                start = int((wav.shape[0] - 25000) / 2)
                stop = int(wav.shape[0] - start)
                print("adding wav[start:stop]: ", wav[start:stop].shape)
                datalist.append(wav[:25000])


                # plt.figure()
                # plt.plot(waveform.t().numpy())
                # plt.show()

            self.trainset = datalist
            self.valset =  datalist

            print('FitLoad Done')

        if step == 'test' or step is None:

            self.testset = 1
            # Load test set
            # self.testset = CIFAR10(os.getcwd(),train=False,
            #         transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, batch_size=self.mb,
                pin_memory=self.pin_memory, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.mb,
                pin_memory=self.pin_memory, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.mb,
                pin_memory=self.pin_memory, num_workers=self.num_workers)



data = SpectrogramDataModule()
data.setup()

print('Starting Data loop')
for datum in data.train_dataloader():
    print(datum.shape)
