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
import argparse
import sys
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import random
import torchvision.models as models


wandb_logger = WandbLogger()
#trainer = Trainer(logger=wandb_logger)
wandb.login()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    #parser.add_argument("train_x", help="The training set input data (npz)")
    #parser.add_argument("train_y", help="The training set target data (npz)")
    #parser.add_argument("dev_x", help="The development set input data (npz)")
    #parser.add_argument("dev_y", help="The development set target data (npz)")

    parser.add_argument("--opt", type=str, \
                        help="The optimizer: \"adadelta\",  \"adagrad\", \"rmsprop\", \"sgd\" "
                             " (string) [default: \"adam\"]", default="adam")
    parser.add_argument("--lr", type=float, \
                        help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("--mb", type=int, \
                        help="The minibatch size (int) [default: 32]", default=16)
    parser.add_argument("-report_freq", type=int, \
                        help="Dev performance is reported every report_freq updates (int) [default: 128]", default=128)
    parser.add_argument("--epochs", type=int, \
                        help="The number of training epochs (int) [default: 10]", \
                        default=10)

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
        file = self.Xlist[index]


        filename = '../../../notdata/task4/' + file


        img = Image.open(filename).convert("RGB")
        tensor = transforms.ToTensor()(img)

        label  = file.split('/')[1]

        yval = 0
        if  (label == 'real'):
            yval = 1


        return tensor, torch.tensor(yval)


class GANFacesDataModule(pl.LightningDataModule):
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


            # grab train data
            trainfilesreal = os.listdir('../../../notdata/task4/train/real')
            trainfilesfake = os.listdir('../../../notdata/task4/train/fake')
            trainfiles = []
            for tfr in trainfilesreal:
                trainfiles.append('train/real/' + tfr)
            for tff in trainfilesfake:
                trainfiles.append('train/fake/' + tff)


            #for file in trainfiles:
                #print(file)


            # grab dev data
            devfilesreal = os.listdir('../../../notdata/task4/dev/real')
            devfilesfake = os.listdir('../../../notdata/task4/dev/fake')
            devfiles = []
            for dfr in devfilesreal:
                devfiles.append('dev/real/' + dfr)
            for dff in devfilesfake:
                devfiles.append('dev/fake/' + dff)



            #for file in devfiles:
                #print(file)

            random.shuffle(trainfiles)
            random.shuffle(devfiles)

            self.trainset = trainfiles
            self.valset =  devfiles

            print('FitLoad Done')

        if step == 'test' or step is None:

            # grab dev data in tuple list
            # grab dev data
            devfilesreal = os.listdir('../../../notdata/task4/dev/real')
            devfilesfake = os.listdir('../../../notdata/task4/dev/fake')
            devfiles = []
            for dfr in devfilesreal:
                devfiles.append('dev/real/' + dfr)
            for dff in devfilesfake:
                devfiles.append('dev/fake/' + dff)



            #for file in devfiles:
                #print(file)

            self.testset = devfiles
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

class LockedDNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=2),
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

        self.log('learning_rate', float(self.lr))
        #print(self.lr)

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

    data = GANFacesDataModule()
    #data.setup()
    #loader = data.train_dataloader()
    #for dat in loader:
    #    print(dat[0].shape, ' y:', dat[1])

    d = 25002
    #model = DeepNeuralNet(d, c, layers)
    #model = LockedDNN()
    resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)

    resnet.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)

    model = LightningWrap(resnet)




    print('start training')

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=[0], logger=wandb_logger)

    print('start fit')
    trainer.fit(model, data)
    print('start test')
    trainer.test(model,datamodule=data)

    torch.save(model.state_dict(), 'task4.pth')

if __name__ == "__main__":
    main(sys.argv)
# python task1.py 2 -lr 0.00025 -mb 16 with 65/35 data split. 10000 samples to 2000 train/dev


# python task1.py 2 -lr 0.00015 gets quick val accur
