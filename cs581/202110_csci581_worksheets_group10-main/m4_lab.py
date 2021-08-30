#! /usr/bin/env python3

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)
An example of building a linear model in PyTorch using nn.Module.
For usage, run with the -h flag.
Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.
"""

import torch
import argparse
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
#or
#from my_classes import Dataset
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
#print(torch.cuda.is_available())
params = {}

#partition = # IDs
#labels = # Labels

#ignore warnings
import warnings
warnings.filterwarnings("ignore")
#interactive mode


n=0
d=0
c=0

class Dataset(torch.utils.data.Dataset):


    def __init__(self, inputPath, labelPath):
        #self.targets = torch.from_numpy(np.load(labelPath).astype(np.float32))
        #self.inputs = torch.from_numpy(np.load(inputPath).astype(np.float32))
        #^ DataLoader will do this for us
        self.targets = np.load(labelPath).astype(np.float32)
        self.inputs = np.load(inputPath).astype(np.float32)
        #self.list_IDs = list_IDs

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X,y

    def D(self):
        return self.inputs.shape[1]
    

class DeepNeuralNet(torch.nn.Module):
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


def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("C", help="The number of classes (int)", \
                        type=int)
    parser.add_argument("train_x", help="The training set input data (npz)")
    parser.add_argument("train_y", help="The training set target data (npz)")
    parser.add_argument("dev_x", help="The development set input data (npz)")
    parser.add_argument("dev_y", help="The development set target data (npz)")

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



def train(model, dataTrain, dataDev, args):

    #train_x = dataTrain.inputs
    #dev_x = dataDev.inputs
    #train_z = dataloader.dataset[]

    #train_y = dataTrain.targets
    #dev_y =  dataDev.targets

    #criterion = loss_func #renaming assignment, criterion = function now.

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = setOpt(model.parameters(), args.lr, args.opt)

    freq = args.report_freq
    batchsize = args.mb

    for epoch in range(args.epochs):
        for update,(mb_x, mb_y) in enumerate(dataTrain):
        #for mb_x, mb_y in dataTrain:
            if (update % freq == 0):
                #evaluate current model on dev set
                numerate = 0
                dominate = 0
                #check minibatch against dev
                for mbdev_x, mbdev_y in dataDev:
                    mbdev_pred = model(mbdev_x)
                    #accuracy with isclose and 0 norm
                    bools = torch.isclose(predicted_class(mbdev_pred), mbdev_y)
                    numRight = np.linalg.norm(bools, 0)

                    numerate += numRight
                    dominate += batchsize

                accuracy = numerate / dominate
                print('Epoch:', epoch, '\t Update: ', update, '\t Accuracy:', accuracy)

            #passing minibatch set to model to get labels
            mb_pred = model(mb_x)
            #mb_y should be truth values
            #print(mb_pred)
            #print(mb_y)
            loss = criterion(mb_pred, mb_y.type(torch.LongTensor))
            loss.backward()
            optimizer.step()  # apply gradients
            optimizer.zero_grad()  # reset the gradient values

    #return tensor of optimal weights
    #print(model.get_weights())

# returning a value [0,1] proportional value of how many are right to total N
def calc_accuracy(ourPredict, ourTruth):

    pred = one_hot(predicted_class(ourPredict), 10)
    return 1 - ((torch.sum(torch.abs(one_hot(predicted_class(pred), 10) - one_hot(ourTruth, 10)))/2) / ourTruth.shape[0])

# returns one hot representation of input tensor x
def one_hot(x, class_count):
    x2 = x.to(torch.int64)
    return torch.eye(class_count)[x2,:]


# returns which class claims one hot label
def predicted_class(classw):
    n, c = classw.shape
    outclass = torch.zeros(n)
    for i in range(n):
        label = torch.argmax(classw[i])
        outclass[i] = label

    return outclass


def main(argv):
    # parse arguments
    args = parse_all_args()

    # load data from our file
    #train_x = torch.from_numpy(np.load(args.train_x).astype(np.float32))
    #train_y = torch.from_numpy(np.load(args.train_y).astype(np.float32))
    #dev_x = torch.from_numpy(np.load(args.dev_x).astype(np.float32))
    #dev_y = torch.from_numpy(np.load(args.dev_y).astype(np.float32))


    #loading from custom dataset class
    dataTrain = Dataset(args.train_x, args.train_y, )
    dataDev   = Dataset(args.dev_x,  args.dev_y)
    
    #training params list
    paramsTrain = {'batch_size': args.mb,
              'shuffle': True,
              'num_workers': 1,
              'drop_last': False}
    #dev params list
    paramsDev = {'batch_size': args.mb,
              'shuffle': False,
              'num_workers': 1,
              'drop_last': False}

    #pass data set
    traingen = torch.utils.data.DataLoader(dataTrain, **paramsTrain)
    devgen = torch.utils.data.DataLoader(dataDev, **paramsDev)


    #grabbing training data
    train_x = dataTrain.inputs
    train_y = dataTrain.targets

    #grabbing dev data
    dev_x = dataDev.inputs
    dev_y = dataDev.targets

    n,d = train_x.shape
    c = 10

    #get args here to pass to our DNN
    layers = args.L

    #splitString = layers.split("x")

    #def __init__(self, inWidth, outWidth, layers):
    model = DeepNeuralNet(d, c, layers)

    #train
    train(model, traingen, devgen, args)

if __name__ == "__main__":
    main(sys.argv)
