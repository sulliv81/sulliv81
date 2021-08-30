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
                             " (string) [default: task4test.txt]", default="task4tests.txt")
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

    soft = torch.nn.Softmax()


    correct = 0
    total = 0
    i = 0
    for dat in loader:

        x = dat
        x.requires_grad_()
        #y = dat[1]
        pred = model(x)
        #print(preds, y)
        #batchacc = acc(preds, y)

        filename = testdata[i]
        label = 'Fake'
        if 'real' in filename:
            label = 'Real'

        score_max_index = pred.argmax()
        score_max_index = 0
        score_max = pred[0,score_max_index]

        confidence = soft(pred[0])[pred.argmax()].detach().numpy()

        zero = torch.tensor(0)
        one = torch.tensor(1)

        fail = False

        if ((label == 'Real') and (pred.argmax().detach().numpy() == 1)) or  ((label == 'Fake') and (pred.argmax().detach().numpy() == 0)):
            correct += 1

        else:
            print('FAIL: ', label, pred.argmax().detach().numpy(), one, zero)
            print(filename)
            fail = True

        if (fail):



            notlabel = 0
            if label == 'Real':
                notlabel = 'Fake'
            else:
                notlabel = 'Real'

            score_max.backward()

            saliency, _ = torch.max(x.grad.data.abs(),dim=1)

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(x.detach().numpy()[0].transpose(1, 2, 0))
            ax[0].axis('off')
            ax[1].imshow(saliency[0], cmap=plt.cm.hot)
            ax[1].axis('off')
            plt.tight_layout()
            fig.suptitle(label + ' Image Labeled as '+ notlabel +', Confidence: ' + str(confidence), size=17)
            plt.savefig('incorrect/' + testlines[i].split('.png')[0].split('test/')[1] + '.png', bbox_inches='tight')
            #plt.show()
            plt.close()
        elif(False) :
            score_max.backward()

            saliency, _ = torch.max(x.grad.data.abs(),dim=1)

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(x.detach().numpy()[0].transpose(1, 2, 0))
            ax[0].axis('off')
            ax[1].imshow(saliency[0], cmap=plt.cm.hot)
            ax[1].axis('off')
            plt.tight_layout()
            fig.suptitle(label + ' Image and Saliency, Confidence: ' + str(confidence), size=17)
            #plt.show()
            plt.savefig('saliency/' + testlines[i].split('.png')[0].split('test/')[1] + '.png', bbox_inches='tight')
            plt.close()







        i += 1

    correct += 0.0000001
    print('ACCURACY: ', correct/i)
    outvec = np.array(resultlist)
    np.savetxt('task4_prediction.npy', outvec)



    #trainer.test(model,datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
