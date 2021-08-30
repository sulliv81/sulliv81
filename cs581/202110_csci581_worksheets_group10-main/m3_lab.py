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

n=0
d=0
c=0


class MultinomialLogReg(torch.nn.Module):
    def __init__(self, D, W, b):
        """
        In the constructor we instantiate our weights and assign them to
        member variables.
        """
        super(MultinomialLogReg, self).__init__()

        self.weights = torch.nn.Parameter(W)
        self.bias = torch.nn.Parameter(b)


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data.
        """

        y_pred = torch.mm(x , self.weights) + self.bias

        sm = torch.nn.Softmax(dim = 1)
        sm_y_pred = sm(y_pred)


        return sm_y_pred

    def get_weights(self):
        return self.weights


def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("D", help="The order of polynomial to fit (int)", \
                        type=int)
    parser.add_argument("train_x", help="The training set input data (npz)")
    parser.add_argument("train_y", help="The training set target data (npz)")
    parser.add_argument("dev_x", help="The development set input data (npz)")
    parser.add_argument("dev_y", help="The development set target data (npz)")
    parser.add_argument("-lambda", type=float, \
                        help="the regularization coefficient (float) [default: 0.0]", default=0.0)

    parser.add_argument("-lr", type=float, \
                        help="The learning rate (float) [default: 0.1]", default=0.1)
    parser.add_argument("-epochs", type=int, \
                        help="The number of training epochs (int) [default: 100]", \
                        default=100)

    return parser.parse_args()

#penalty = lambda, which is regularization coefficient
# don't want to regularize b, can pass more args
# hopefully returning frob norm^2
def loss_func(ypred, truey, w, pen):
    # torches builtin Cross Entropy Loss feature for our multiLogReg (many classes)
    CE = torch.nn.CrossEntropyLoss()
    #frobenius normalization
    return CE(ypred, truey.type(torch.LongTensor)) + (pen * torch.norm(w, 'fro'))

def train(model, train_x, train_y, dev_x, dev_y, args):
    # define our loss function

    criterion = loss_func #renaming assignment, criterion = function now.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    for epoch in range(args.epochs):
        # make prediction
        y_pred = model(train_x)

        # compute loss
        loss = criterion(y_pred, train_y, model.get_weights(), getattr(args,'lambda'))  # compute loss

        #get train accuracy
        train_acc = calc_accuracy(y_pred, train_y)

        # take gradient step
        optimizer.zero_grad()  # reset the gradient values
        loss.backward()  # compute the gradient values
        optimizer.step()  # apply gradients

        # eval on dev
        dev_y_pred = model(dev_x)
        dev_acc = calc_accuracy(dev_y_pred, dev_y)

        #aggregate accuracy print outs
        print("train accuracy = %.3f, dev accuracy = %.3f" % (train_acc, dev_acc))

    #return tensor of optimal weights
    print(model.get_weights())

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

    # load data
    train_x = torch.from_numpy(np.load(args.train_x).astype(np.float32))
    train_y = torch.from_numpy(np.load(args.train_y).astype(np.float32))
    dev_x = torch.from_numpy(np.load(args.dev_x).astype(np.float32))
    dev_y = torch.from_numpy(np.load(args.dev_y).astype(np.float32))

    n,d = train_x.shape
    c = 10
    model = MultinomialLogReg(args.D, torch.zeros((d, c)), torch.zeros(c))

    train(model, train_x, train_y, dev_x, dev_y, args)


if __name__ == "__main__":
    main(sys.argv)
