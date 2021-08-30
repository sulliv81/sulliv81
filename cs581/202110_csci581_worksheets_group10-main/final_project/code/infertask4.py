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
from PIL import Image



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
                             " (string) [default: ../../../notdata/task4/test]", default="../../../notdata/task4/test")
    parser.add_argument("-txt", type=str, \
                        help="The file path to the testing txt file "
                             " (string) [default: task4test.txt]", default="task4.script.txt")
    parser.add_argument("-mod", type=str, \
                        help="Path to the saved model state dict "
                             " (string) [default: task4.pth]", default="task4mega.pth")


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

        #print('File: ', file)
        filename = '../../../notdata/task4/' + file


        img = Image.open(filename).convert("RGB")
        tensor = transforms.ToTensor()(img)

        #label  = file.split('/')[1]

        #yval = 0
        #if  (label == 'real'):
            #yval = 1


        return tensor#, #torch.tensor(yval)


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

            testfile = open(args.txt, 'r')
            testlines = testfile.readlines()

            testdata  = []
            for line in testlines:
                if (line.rstrip() != ''):
                    testdata.append(line.rstrip())

                #print(file1, ' ', file2)

                #traindata.append((file1, file2))
                # plt.figure()
                # plt.plot(waveform.t().numpy())
                # plt.show()



            #for file in devfiles:
                #print(file)

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


    testpath = args.tp
    data = GANFacesDataModule()
    data.setup()
    loader = data.test_dataloader()

    resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)

    model = LightningWrap(resnet)
    model.load_state_dict(torch.load(args.mod))
    model.eval()

    acc = pl.metrics.Accuracy()
    resultlist = []

    testfile = open(args.txt, 'r')
    testlines = testfile.readlines()

    testdata  = []
    for line in testlines:
        if (line.rstrip() != ''):
            testdata.append(line.rstrip())


    correct = 0
    total = len(loader)
    i = 0
    for dat in loader:

        x = dat
        #y = dat[1]
        preds = model(x)
        #print(preds, y)
        #batchacc = acc(preds, y)
        for pred in preds:
            filename = testdata[i]
            # label = 'fake'
            # if 'real' in filename:
            #     label = 'real'

            #print(label)
            pred = nn.Softmax()(pred)
            #print(pred.shape)
            prob1 = 1 -  pred[1].detach().numpy()
            resultlist.append(prob1)
            if i % 10 == 0:
                print(prob1)
                print(i, ' / ', total)
            #total += 1
            # if ((label == 'real') and (np.round(prob1) == 0)):
            #     #print('..')
            #     correct += 1
            #     print("correct")
            # elif ((label == 'fake') and (np.round(prob1) == 1)):
            #     #print('.')
            #     correct += 1
            #     print("correct")
            # else:
            #     print('FAIL')
            #     print(filename)



            i += 1


    #print('ACCURACY: ', correct/total)
    outvec = np.array(resultlist)
    np.savetxt('task4_predictions.npy', outvec)



    #trainer.test(model,datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
