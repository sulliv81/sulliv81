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
import sys
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import StepLR
import random



def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    #parser.add_argument("C", help="The number of classes (int)", \
                        #type=int)
    #parser.add_argument("train_x", help="The training set input data (npz)")
    #parser.add_argument("train_y", help="The training set target data (npz)")
    #parser.add_argument("dev_x", help="The development set input data (npz)")
    #parser.add_argument("dev_y", help="The development set target data (npz)")

    parser.add_argument("-tp", type=str, \
                        help="The file path to the testing folder "
                             " (string) [default: ../../../notdata/task1/test]", default="../../../notdata/task1/test")
    parser.add_argument("-txt", type=str, \
                        help="The file path to the testing txt pairs file "
                             " (string) [default: devspeakerpairs.txt]", default="task1.script.txt")
                             #" (string) [default: devspeakerpairs.txt]", default="devspeakerpairs.txt")
    parser.add_argument("-mod", type=str, \
                        help="Path to the saved model state dict "
                             " (string) [default: task1.pth]", default="task1march5th.pth")


    parser.add_argument("-opt", type=str, \
                        help="The optimizer: \"adadelta\",  \"adagrad\", \"rmsprop\", \"sgd\" "
                             " (string) [default: \"adam\"]", default="adam")

    parser.add_argument("-lr", type=float, \
                        help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-clip", type=float, \
                        help="The clipping length of audio files [default: 25000]", default=25000)
    parser.add_argument("-mb", type=int, \
                        help="The minibatch size (int) [default: 32]", default=1)

    parser.add_argument("-epochs", type=int, \
                        help="The number of training epochs (int) [default: 100]", \
                        default=100)


    return parser.parse_args()

args = parse_all_args()



class NotADataset(Dataset):
    #Dataset to index the list and be used with our data loader
    def __init__(self, datalist):
        self.Xlist = datalist
        self.length = len(datalist)
        self.cliplen = args.clip
        #self.data_type = type
        #self.seq_len = seq_len
        #self.num_speakers = len(self.Xlist)
    #returns total indicies
    def __len__(self):
        return self.length
    #returns specific index
    #next step load each file indicually
    def __getitem__(self, index):
        file1 = self.Xlist[index][0]
        file2 = self.Xlist[index][1]

        file1name = '../../../notdata/task1/' + file1
        file2name = '../../../notdata/task1/' + file2

        waveform1, sample_rate1 = torchaudio.load(file1name)
        waveform2, sample_rate2 = torchaudio.load(file2name)


        wav1 = waveform1.flatten()
        if (wav1.shape[0] > self.cliplen):

            start1 = int((wav1.shape[0] - self.cliplen) / 2)
            start1 = random.randint(0, np.abs(start1))
            stop1 = int(wav1.shape[0] - start1)
        else:
            start1 = 0
            stop1 = wav1.shape[0]

        wav2 = waveform2.flatten()
        if (wav2.shape[0] > self.cliplen):

            start2 = int((wav2.shape[0] - self.cliplen) / 2)
            start2 = random.randint(0, np.abs(start2))
            stop2 = int(wav2.shape[0] - start2)
        else:
            start2 = 0
            stop2 = wav1.shape[0]
        #print("adding wav[start:stop]: ", wav[start:stop][:25000].shape)

        #extract file nums from file names to append number to right path


        speaker1FileNo = int(file1.split('.wav')[0].split('file')[1])
        speaker2FileNo = int(file2.split('.wav')[0].split('file')[1])

        speaker1str = file1.split('.wav')[0]
        speaker2str = file2.split('.wav')[0]

        #speaker1 = int(np.loadtxt('../../../notdata/task1/' + speaker1str + '.spkid.txt'))
        #speaker2 = int(np.loadtxt('../../../notdata/task1/' + speaker2str + '.spkid.txt'))

        yval = 0
        #if  (speaker1 == speaker2):
        #    yval = 1


        #print('SPK1: ', speaker1FileNo, 'SPK2: ', speaker2FileNo, 'yval: ', yval)
        xchunk1 = wav1[start1:stop1][:self.cliplen]
        xchunk2 = wav2[start2:stop2][:self.cliplen]

        if (xchunk1.shape[0] < self.cliplen):
            #print("Keleven")
            #xchunk1 = torch.tensor(np.zeros(25000), dtype=torch.float32)
            while(xchunk1.shape[0] < self.cliplen):
                xchunk1 = torch.hstack((xchunk1, xchunk1))

            xchunk1 = xchunk1[:self.cliplen]

        if (xchunk2.shape[0] < self.cliplen):
            #print("Keleven2")
            #xchunk2 = torch.tensor(np.zeros(25000), dtype=torch.float32)
            while(xchunk2.shape[0] < self.cliplen):
                xchunk2 = torch.hstack((xchunk2, xchunk2))

            xchunk2 = xchunk2[:self.cliplen]



        specgram1  = torchaudio.transforms.Spectrogram()(xchunk1)
        specgram2  = torchaudio.transforms.Spectrogram()(xchunk2)

        deepstack = torch.zeros((2, specgram1.shape[0], specgram1.shape[1]))

        deepstack[0] = specgram1
        deepstack[1] = specgram2
        #deepstack[2] = specgram1 - specgram2

        #decide how to return both xchunk1/2
        #step one, try vstack
        doubleStacker = deepstack#torch.hstack((specgram1, specgram2)).unsqueeze(0)

        #print("double stack size: " , doubleStacker.shape)
        return doubleStacker, torch.tensor(yval)


class SpectrogramDataModule(pl.LightningDataModule):
    def __init__(self,pin_memory=False,num_workers=4,val_n=5000):
        super().__init__()

        self.mb = args.mb
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_n = val_n

        #self.transform=transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        print('No data to Prepare')
        # download data if not downloaded
        #CIFAR10(os.getcwd(),train=True,download=True)
        #CIFAR10(os.getcwd(),train=False,download=True)

    def setup(self,step=None):
        # make splits, create Dataset objects

        if step == 'fit' or step is None:


            #filestrain = np.loadtxt('task1traindata.txt', dtype='int')

            #filesdev = np.loadtxt('task1devdata.txt', dtype='int')

            # grab train data in tuple list
            trainPairs = open('trainspeakerpairs.txt', 'r')
            trainlines = trainPairs.readlines()

            traindata  = []
            for line in trainlines:
                bothFiles = line.split(' ')
                file1 = bothFiles[0]
                file2 = bothFiles[1].rstrip()

                #print(file1, ' ', file2)

                traindata.append((file1, file2))
                # plt.figure()
                # plt.plot(waveform.t().numpy())
                # plt.show()


            #grab dev data in tuple list
            devPairs = open('devspeakerpairs.txt', 'r')
            Devlines = devPairs.readlines()

            devdata = []
            for line in Devlines:
                bothFiles = line.split(' ')
                file1 = bothFiles[0]
                file2 = bothFiles[1].rstrip()

                #print(file1, ' ', file2)

                devdata.append((file1, file2))

            self.trainset = traindata



            self.valset =  devdata

            print('FitLoad Done')

        if step == 'test' or step is None:

            # grab dev data in tuple list
            testPairs = open(args.txt, 'r')
            testlines = testPairs.readlines()

            testdata = []
            for line in testlines:
                bothFiles = line.split(' ')
                file1 = bothFiles[0]
                file2 = bothFiles[1].rstrip()

                #print(file1, ' ', file2)

                testdata.append((file1, file2))

            self.testset = testdata
            # Load test set
            # self.testset = CIFAR10(os.getcwd(),train=False,
            #         transform=self.transform)


    def train_dataloader(self):
        train_dataset = NotADataset(self.trainset)
        train_loader = DataLoader(train_dataset,
                                  shuffle = True,
                                  batch_size=self.mb,
                                  num_workers = self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = NotADataset(self.valset)
        val_loader = DataLoader(val_dataset,
                                shuffle = False,
                                batch_size=self.mb,
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = NotADataset(self.testset)
        test_loader = DataLoader(test_dataset,
                                 shuffle = False,
                                 batch_size=self.mb,
                                 num_workers = self.num_workers)

        return test_loader



class LockedDNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

        self.lr = args.lr
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        #batch_size,_,_,_ = x.size()
        #x = x.view(batch_size, -1)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x



    def eval_batch(self, batch, batch_idx):
        # Make predictions
        x, y = batch
        y_pred = self(x)

        # Evaluate predictions
        loss = F.cross_entropy(y_pred, y)
        acc = self.accuracy(y_pred, y)
        #print(acc, y)
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

        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]


def main(argv):
    # parse arguments


    testpath = args.tp
    data = SpectrogramDataModule()
    data.setup()
    loader = data.test_dataloader()

    model = LockedDNN()
    model.load_state_dict(torch.load(args.mod))
    model.eval()

    #total = 0.0000001
    #correct = 0.0000001

    acc = pl.metrics.Accuracy()
    resultlist = []
    for dat in loader:
        x = dat[0]
        y = dat[1]
        preds = model(x)
        #print(preds, y)
        #batchacc = acc(preds, y)
        for pred in preds:
            pred = nn.Softmax()(pred)
            prob1 = pred[1].detach().numpy()
            resultlist.append(prob1)
            print(prob1, y)
            if (np.round(prob1) == 1):
                print("match")
            #total += 1
            #if (np.round(prob1) == y):
                #correct += 1



    outvec = np.array(resultlist)
    np.savetxt('task1_predictions_do.npy', outvec)
    #print('ACC: ', correct / total)



    #trainer.test(model,datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
