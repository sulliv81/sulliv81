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
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
torchaudio.set_audio_backend = "sox_io"
torch.set_printoptions(edgeitems=1)

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

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
            #xchunk1 = torch.tensor(np.zeros(25000), dtype=torch.float32)
            while(xchunk1.shape[0] < 25000):
                xchunk1 = torch.hstack((xchunk1, xchunk1))

            xchunk1 = xchunk1[:25000]

        if (xchunk2.shape[0] < 25000):
            #print("Keleven2")
            #xchunk2 = torch.tensor(np.zeros(25000), dtype=torch.float32)
            while(xchunk2.shape[0] < 25000):
                xchunk2 = torch.hstack((xchunk2, xchunk2))

            xchunk2 = xchunk2[:25000]

        spectro =  torchaudio.transforms.Spectrogram()

        specgram1  = spectro(xchunk1).unsqueeze(0)
        specgram2  = spectro(xchunk2).unsqueeze(0)

        specgram1_r = F.interpolate(specgram1, size=224)[0]
        specgram2_r = F.interpolate(specgram2, size=224)[0]

        deepstack = torch.zeros((2, specgram1_r.shape[0], specgram1_r.shape[1]))

        deepstack[0] = specgram1_r
        deepstack[1] = specgram2_r
        #deepstack[2] = torch.zeros((specgram1_r.shape[0], specgram1_r.shape[1]))
        normalize = transforms.Normalize(mean=[0.485, 0.456],
                                 std=[0.229, 0.224])


        #decide how to return both xchunk1/2
        #step one, try vstack
        tripleStacker = normalize(deepstack)#torch.hstack((specgram1, specgram2)).unsqueeze(0)

        #print("double stack size: " , doubleStacker.shape)
        return tripleStacker, torch.tensor(yval)


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

        self.dropout = nn.Dropout(.1)

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
        print(x.shape)
        #layer activation loop
        for i in range(numlayers):
            #pass through f(z)
            x = self.linears[i](x)
            x = self.dropout(x)
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
        self.log('train_acc', acc, prog_bar=True)

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

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

class LockedDNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=4),
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

        # self.cnn1 = nn.Sequential(
        # nn.Conv2d(1, 96, kernel_size=11,stride=1),
        # nn.ReLU(inplace=True),
        # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
        # nn.MaxPool2d(3, stride=2),
        #
        # nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
        # nn.ReLU(inplace=True),
        # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
        # nn.MaxPool2d(3, stride=2),
        # nn.Dropout2d(p=0.3),
        #
        # nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(3, stride=2),
        # nn.Dropout2d(p=0.3),
        # )
        #
        #   # Defining the fully connected layers
        # self.fc1 = nn.Sequential(
        #   # First Dense Layer
        # nn.Linear(30976, 1024),
        # nn.ReLU(inplace=True),
        # nn.Dropout2d(p=0.5),
        #   # Second Dense Layer
        # nn.Linear(1024, 128),
        # nn.ReLU(inplace=True),
        #   # Final Dense Layer
        # nn.Linear(128,2)
        # )

        self.lr = args.lr
        self.accuracy = Accuracy()
        self.loss = ContrastiveLoss(.5)

    def forward(self, x):
        #batch_size,_,_,_ = x.size()
        #x = x.view(batch_size, -1)
        #print('X Shape:', x.shape)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        # x = self.cnn1(x)
        # x = x.view(x.size()[0], -1)
        # x = self.fc1(x)
        return x



    def eval_batch(self, batch, batch_idx):
        # Make predictions
        x, y = batch
        y_pred1 = self(x[:,0].unsqueeze(1))
        y_pred2 = self(x[:,1].unsqueeze(1))

        # Evaluate predictions
        loss = self.loss(y_pred1, y_pred2, y)

        dist = F.pairwise_distance(y_pred1, y_pred2)
        #acc = self.accuracy(dist, y)
        pred = torch.where(dist > .5, 0, 1)
        print('\n', dist, '\n', y)


        acc = self.accuracy(pred, y)
        #print(dist, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)

        x,y = batch
        y_pred1 = self(x[:,0].unsqueeze(1))
        y_pred2 = self(x[:,1].unsqueeze(1))

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


class LightningWrap(pl.LightningModule):

    def __init__(self, model):
        super(LightningWrap, self).__init__()
        self.model = model
        self.criteria = nn.CrossEntropyLoss()
        self.lr = args.lr
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)


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
        self.log('train_acc', acc, prog_bar=True)

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
    #for dat in loader:
    #    print(dat[0].shape, ' y:', dat[1])

    d = 25002
    layers = args.L
    #model = DeepNeuralNet(d, c, layers)
    model = LockedDNN()
    #resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
    #model = LightningWrap(resnet)


    print('start training')

    wandb_logger = WandbLogger()
    #trainer = Trainer(logger=wandb_logger)

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=[0], logger=wandb_logger)

    print('start fit')
    trainer.fit(model, data)
    print('start test')
    trainer.test(model,datamodule=data)

if __name__ == "__main__":
    main(sys.argv)
