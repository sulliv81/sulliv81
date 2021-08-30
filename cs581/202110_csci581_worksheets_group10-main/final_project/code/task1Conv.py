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
import sys
import torch.fft


torchaudio.set_audio_backend = "sox_io"

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("C", help="The number of classes (int)", \
                        type=int)
    #parser.add_argument("train_x", help="The training set input data (npz)")
    #parser.add_argument("train_y", help="The training set target data (npz)")
    #parser.add_argument("dev_x", help="The development set input data (npz)")
    #parser.add_argument("dev_y", help="The development set target data (npz)")

    parser.add_argument("-f1", type=str, \
                        help="The hidden activation function: \"relu\" or \"tanh\" or \"sigmoid\" "
                             " (string) [default: \"relu\"]", default="relu")
    parser.add_argument("-opt", type=str, \
                        help="The optimizer: \"adadelta\",  \"adagrad\", \"rmsprop\", \"sgd\" "
                             " (string) [default: \"adam\"]", default="adam")
    parser.add_argument("-L", type=str, \
                        help="A comma delimited  list of nunits by nlayers specifiers (see assignment pdf) (string) "
                             "[dafault: \"32x1\"]", default="32x1")
    parser.add_argument("-lr", type=float, \
                        help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-mb", type=int, \
                        help="The minibatch size (int) [default: 32]", default=32)
    parser.add_argument("-report_freq", type=int, \
                        help="Dev performance is reported every report_freq updates (int) [default: 128]", default=128)
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
        self.drop = nn.Dropout(0.2)
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
        start1 = int((wav1.shape[0] - 25000) / 2)
        stop1 = int(wav1.shape[0] - start1)

        wav2 = waveform2.flatten()
        start2 = int((wav2.shape[0] - 25000) / 2)
        stop2 = int(wav2.shape[0] - start2)
        #print("adding wav[start:stop]: ", wav[start:stop][:25000].shape)

        #extract file nums from file names to append number to right path


        speaker1FileNo = int(file1.split('.wav')[0].split('file')[1])
        speaker2FileNo = int(file2.split('.wav')[0].split('file')[1])

        speaker1str = file1.split('.wav')[0]
        speaker2str = file2.split('.wav')[0]

        speaker1 = int(np.loadtxt('../../../notdata/task1/' + speaker1str + '.spkid.txt'))
        speaker2 = int(np.loadtxt('../../../notdata/task1/' + speaker2str + '.spkid.txt'))

        yval = 0
        if  (speaker1 == speaker2):
            yval = 1


        #print('SPK1: ', speaker1FileNo, 'SPK2: ', speaker2FileNo, 'yval: ', yval)
        xchunk1 = wav1[start1:stop1][:25000]
        xchunk2 = wav2[start2:stop2][:25000]

        if (xchunk1.shape[0] < 25000):
            #print("Keleven")
            xchunk1 = torch.tensor(np.zeros(25000), dtype=torch.float32)

        if (xchunk2.shape[0] < 25000):
            #print("Keleven2")
            xchunk2 = torch.tensor(np.zeros(25000), dtype=torch.float32)


        #fft1 = torch.fft.rfft(xchunk1, 25000, dim=0, norm=None).float()
        #print(fft1.shape)
        #fft2 = torch.fft.rfft(xchunk2, 25000, dim=0, norm=None).float()

        #print(fft1.shape)

        #decide how to return both xchunk1/2
        #step one, try vstack
        #doubleStacker = torch.hstack((fft1, fft2))

        t = torch.zeros((25000, 2))

        #("t shape: ", t.shape)
        t[:,0] = xchunk1
        t[:,1] = xchunk2
        doubleStacker = t
        #doubleStacker = torch.hstack((xchunk1, xchunk2))
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
            trainPairs = open('trainpairs.txt', 'r')
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
            devPairs = open('devpairs.txt', 'r')
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
            devPairs = open('devpairs.txt', 'r')
            Devlines = devPairs.readlines()

            devdata = []
            for line in Devlines:
                bothFiles = line.split(' ')
                file1 = bothFiles[0]
                file2 = bothFiles[1].rstrip()

                #print(file1, ' ', file2)

                devdata.append((file1, file2))

            self.testset = devdata
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

class DeepNeuralNet(pl.LightningModule):
    def __init__(self, inWidth, outWidth, layers):
        """
        In the constructor we instantiate our weights and assign them to
        member variables.
        """
        super(DeepNeuralNet, self).__init__()

        self.linears = nn.ModuleList()

        #setting up  hidden layer size through arg parse in our initialization
        commaSplit = layers.split(",")
        firstChangeSize = inWidth

        # set up relu activation in init
        self.relu = nn.ReLU()

        #set up softmax post act
        self.softmax = nn.Softmax()

        for z in commaSplit:
            splitString = z.split("x") #"32x4,64x2" --> {32x4 , 64x2}
            hiddenLayerSize = int(splitString[0] )   #grabbing size
            numUnits = int(splitString[1])    #grabbing numUnits

            #now define hidden layer sizes
            flag1 = 0

            #hidden layers
            for i in range(numUnits):
                if (flag1 == 0):
                    flag1 = 1
                    self.linears.append(nn.Linear(firstChangeSize, hiddenLayerSize)) #creates layer
                    firstChangeSize = hiddenLayerSize #update next change size
                else:
                    self.linears.append(nn.Linear(hiddenLayerSize, hiddenLayerSize))

            #output dimensions
        self.linears.append((nn.Linear(firstChangeSize, outWidth)))

        self.lr = args.lr

        self.accuracy = Accuracy()
        #print("b4 params")
        #print params
        for name, param in self.named_parameters():
            print(name, param.data.shape)
        #print("after params for loop")


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data.
        """

        #y_pred = torch.mm(x , self.weights) + self.bias
        #print('forward')
        #sm = torch.nn.Softmax(dim = 1)
        #sm_y_pred = sm(y_pred)
        numlayers = len(self.linears)
        #layer activation loop
        for i in range(numlayers):
            #pass through f(z)
            x = self.linears[i](x)

            #apply activation, last layer gets softmax
            if (i == (numlayers - 1)):
                x = self.softmax(x)
            else:
                x = self.relu(x)

        return x


       # for (i, x) in enumerate(self.linears[0], self.linears.len-1):
          #  x = F.relu(self.linears(x))
        #softmax last one
       # x = self.linears.len-1(x)

        #return F.softmax(x, dim=1)

    def get_linears(self):
        return self.linears

    def eval_batch(self, batch, batch_idx):
        #print('eval')
        # Make predictions
        x, y = batch
        y_pred = self(x)




        # Evaluate predictions
        loss = F.cross_entropy(y_pred, y)


        acc = self.accuracy(y_pred, y)
        #print(acc, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        #print('train')
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

class LockedDNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #conv
        self.conv1 = nn.Conv2d(1,1,(2,2),stride=1,padding=0)
        self.pool = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(24999, 5000)
        self.fc2 = nn.Linear(5000, 2500)
        self.fc3 = nn.Linear(2500, 350)

        self.lr = args.lr

        self.accuracy = Accuracy()

    def forward(self, x):
        #batch_size,_,_,_ = x.size()
        #x = x.view(batch_size, -1)

        #apply conv
        #print("x shape: ", x.shape)
        #x = self.conv1(x)
        x= self.pool(x)
        #x shape:  torch.Size([1, 12500, 1])
        #print("x shape: ", x.shape)
        #print("view flat shape: ", x.view((x.shape[0], 12500)).shape)
        x = self.fc1(x.view((x.shape[0], 24999)))
        x = self.fc2(x)
        x = self.fc3(x)

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
        return


def main(argv):
    # parse arguments

    data = SpectrogramDataModule()
    data.setup()
    loader = data.train_dataloader()
    for dat in loader:
        print(dat[0].shape, ' y:', dat[1])

    d = 12501
    c = args.C
    layers = args.L
    model = LockedDNN()
    #model = LockedDNN()


    print('start training')

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=[0])

    print('start fit')
    trainer.fit(model, data)
    print('start test')
    trainer.test(model,datamodule=data)

if __name__ == "__main__":
    main(sys.argv)
