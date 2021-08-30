#! /usr/bin/env python3

"""
@authors: Brian Hutchinson

A simple speaker embedding model based on:

Yair Movshovitz-Attias, Alexander Toshev, Thomas K. Leung, Sergey Ioffe and Saurabh Singh. No Fuss Distance Metric Learning using Proxies. arXiv:1703.07464, 2017.

For usage, run with the -h flag.

Disclaimers:
- Distributed as-is.
- Please contact me if you find any issues with the code.

"""

import torch
import argparse
import sys
import numpy as np
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from pytorch_lightning.metrics import Accuracy
import pytorch_lightning as pl


colors = np.random.rand(10,3)



def parse_all_args():
    """Parses all arguments.

        Returns:
            argparse.Namespace: the parsed argument object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", help="The location of the acoustic features files (string)")
    parser.add_argument("train_list", help="A file with the IDs of the train set speakers (string)")
    parser.add_argument("dev_list", help="A file with the IDs of the dev set speakers (string)")

    parser.add_argument("-L", type=int, \
                        help="The hidden layer dimension (int) [default: 128]", default=128)
    parser.add_argument("-K", type=int, \
                        help="The of stacked LSTMs (int) [default: 2]", default=2)
    parser.add_argument("-nframes", type=int, \
                        help="The length of audio used to produce embedding, in frames (int) [default: 200]",
                        default=200)
    parser.add_argument("-opt", \
                        help="The optimizer: \"adadelta\", \"adagrad\", \"adam\", \"rmsprop\", \"sgd\" (string) [default: \"adam\"]",
                        default="adam")
    parser.add_argument("-lr", type=float, \
                        help="The learning rate (float) [default: 0.01]", default=0.01)
    parser.add_argument("-updates", type=int, \
                        help="The number of training updates (int) [default: 50000]", default=50000)

    return parser.parse_args()

args = parse_all_args()





class TimeseriesDataset(Dataset):
    #Dataset to index the list and be used with our data loader
    def __init__(self, datalist, seq_len: int = 20):
        self.Xlist = datalist
        self.seq_len = seq_len
        self.num_speakers = len(self.Xlist)
    #returns total indicies
    def __len__(self):
        lensum = 0
        for i in range(len(self.Xlist)):
            lensum += self.Xlist[i].shape[0] - (self.seq_len-1)
        return lensum
    #returns specific index
    def __getitem__(self, index):
        findindex = index
        for i in range(self.num_speakers):
            filen = self.Xlist[i].shape[0] - (self.seq_len-1)
            #in this file
            if (findindex < filen):
                #print('DATA: ', self.Xlist[i][findindex:findindex+self.seq_len].shape, ' iter: ', i)
                #return (self.Xlist[i][findindex:findindex+self.seq_len, :],  (np.ones(self.seq_len) * i).astype(np.compat.long))
                return (self.Xlist[i][findindex:findindex+self.seq_len, :],  torch.tensor([i], dtype=torch.long))
            else:
                findindex -= filen
                #subtract of this file's indecies

        # find y in here given index




#LightningModule for TinyTimit
class TinyTimitDataModule(pl.LightningDataModule):
    def __init__(self,pin_memory=False,num_workers=6,val_n=5000, seq_len = 200):
        super().__init__()

        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_n = val_n
        self.seq_len = seq_len

        self.transform=transforms.Compose([transforms.ToTensor()])
        #hard copy for use with visualiztion
        self.trainset = []


    def setup(self,step=None):
        # make splits, create Dataset objects

        # Load test set
        dev_set = []
        with open(args.dev_list, 'r') as f:
            for line in f:
                fn = "%s/%s.txt" % (args.data_dir, line.rstrip())
                print("Attempting to load [%s]" % fn)
                next_x = np.genfromtxt(fn, delimiter=' ', dtype=np.float32)
                dev_set.append(torch.tensor(next_x))

        if step == 'fit' or step is None:
            # Load train and val
            train_set = []

            with open(args.train_list, 'r') as f:
                for line in f:
                    fn = "%s/%s.txt" % (args.data_dir, line.rstrip())
                    print("Attempting to load [%s]" % fn)
                    next_x = np.genfromtxt(fn, delimiter=' ', dtype=np.float32)
                    train_set.append(torch.tensor(next_x))



            self.trainset = train_set
            self.valset = dev_set

        if step == 'test' or step is None:

            self.testset = dev_set




    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.trainset, seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset,
                                  shuffle = True,
                                  num_workers = self.num_workers)

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.valset, seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset,
                                shuffle = False,
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.testset, seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset,
                                 shuffle = False,
                                 num_workers = self.num_workers)

        return test_loader

    #Hard copy getter
    def getTrainSet(self):
        train_set = []

        with open(args.train_list, 'r') as f:
            for line in f:
                fn = "%s/%s.txt" % (args.data_dir, line.rstrip())
                print("Attempting to load [%s]" % fn)
                next_x = np.genfromtxt(fn, delimiter=' ', dtype=np.float32)
                train_set.append(torch.tensor(next_x).cuda())
        return train_set



class SpeakerEmbedding(pl.LightningModule):
    def __init__(self, D, Ntr, args, trainsetter):
        """Instantiates all model parameters.

        Args:
            D (int): input dimension
            Ntr (int): number of speaker in training set
            args (argparse.Namespace): arguments object
        """
        super(SpeakerEmbedding, self).__init__()
        self.lr = args.lr
        self.accuracy = Accuracy()
        self.D   = D
        self.Ntr = Ntr
        self.enc_lstm = torch.nn.LSTM(D,args.L,args.K,batch_first=True,bidirectional=True)

        self.proxies = torch.nn.Parameter(torch.zeros((Ntr,4*args.L)))
        torch.nn.init.xavier_uniform_(self.proxies)

        self.dist = torch.nn.PairwiseDistance()
        self.idx = 0
        self.trainsethardcopy = trainsetter
        self.optim = torch.nn.CrossEntropyLoss()

        # print model parameters as a sanity check
        for name, param in self.named_parameters():
            print(name,param.data.shape)

    def forward(self, x):
        """Maps a single sequence to a fixed length vector.

        Args:
            x (torch.Tensor): input sequence (TxD)
            args (argparse.Namespace): arguments object

        Returns:
            torch.Tensor: speaker embedding (1x4L)
        """
        # N files
        # F nframes
        # T length of ith file

        # output should be 1xTx2L
        #print('XSHAPE: ', x.shape)
        #x = x[0]
        output, _ = self.enc_lstm(x.reshape((1,-1,self.D)))

        output_fwd, output_back = torch.chunk(output, 2, 2) # separate fwd and back

        combo_out = torch.cat([output_fwd[:,-1,:],output_back[:,0,:],torch.mean(output,dim=1).squeeze(dim=1)],dim=1) # 4L

        return combo_out

    def compute_loss(self,criterion,x,i):
        """Computes proxy loss value.

        Args:
            criterion: cross-entropy loss object
            x (torch.Tensor): embedding of speaker (1x4L)
            i (torch.Tensor): speaker id in {0,1,...,Ntr-1} (scalar)

        Returns:
            torch.Tensor: loss (scalar) torch.repeat repeats along axis
        """

        #ndists = -self.dist(x.repeat(self.Ntr,1),self.proxies)
        #print('XBEFORE', x.shape)
        ndists = -self.dist(x.repeat(self.Ntr,1),self.proxies)
        #ndists = -self.dist(x,self.proxies)
        #print('THE TRUTH WILL SET YOU FREE: ', i)
        #print('PRED SHAPE: ', ndists.reshape(1,-1).shape)
        #print('PRED SHAPE: ', ndists.reshape(1,-1).shape)
        loss = criterion(ndists.reshape(1,-1), i.reshape(1))
        return loss, 0

    def training_step(self, batch, batch_idx):
        loss,acc = self.eval_batch(batch,batch_idx)

        x,y = batch
        y_pred = self(x[0])

        self.log('train_loss', loss)
        #self.log('train_acc', acc)

        #print(batch_idx)


        return loss

    def eval_batch(self, batch, batch_idx):
        # Make predictions
        x, y = batch
        y_pred = self(x[0])

        # Evaluate predictions
        #def compute_loss(self, criterion, x, i):
        loss = self.compute_loss(self.optim, y_pred, y)

        #print('ACCURACY PRED:', y_pred.repeat(self.Ntr,1).reshape(1,-1).shape)
        #print('ACCURACY TRUE:', y[0][0])
        #acc = self.accuracy(y_pred, y[0][0])
        #acc = .6

        return loss
    #Hijack validation step to do visualiztion
    def validation_step(self, batch, batch_idx):
        loss = self.eval_batch(batch,batch_idx)
        self.log('val_loss', loss)

        if (batch_idx % 5000 == 0):
            print('Report: ', self.idx)

            train_set = self.trainsethardcopy
            #print(train_set)
            embeddings,labels = embed_set(20, self , train_set ,args) # hardcoded magic number :(
            neighbor_report(embeddings,labels,self)
            visualize_embeddings(embeddings,labels,colors,self.idx)
            self.idx += 1




    def test_step(self, batch, batch_idx):
        loss = self.eval_batch(batch,batch_idx)
        self.log('test_loss', loss)


    def configure_optimizers(self):

        optVal = args.opt
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if (optVal == "adam"):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif (optVal == "adagrad"):
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        elif (optVal == "rmsprop"):
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif (optVal == "sgd"):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif (optVal == "adadelta"):
            optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr)


        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return [optimizer], [scheduler]



def embed_set(nsamples,model,dataset,args):
    """Embeds all sequences in a dataset.  Used for visualization and analysis.

    Args:
        nsamples (int): the number of T length audio snippets to randomly extract from each speaker
        model (SpeakerEmbedding object): a model
        dataset (list): a list of TxD torch.Tensor objects containing input sequences; one for each speaker in dataset (length N)
        args (argparse.Namespace): arguments object

    Returns:
        torch.Tensor,list: embeddings for each snippet extracted (nsamples*Nx4L), nsamples*N corresponding speaker ID labels
    """
    embeddings = []
    labels     = []

    # embed data
    for idx,speaker in enumerate(dataset):
        for i in range(nsamples):
            #print('SPEAKER: ', datax[0].shape )
            sframe = math.floor(random.random()*(speaker.shape[0]-args.nframes))
            mb_x = speaker[sframe:(sframe+args.nframes),:]
            #print('MBX: ', mb_x.shape)
            #print(mb_x.reshape((1, mb_x.shape[0], mb_x.shape[1])).shape)
            embedding = model(mb_x.reshape((1, mb_x.shape[0], mb_x.shape[1])))
            embeddings.append(embedding)
            labels.append(idx)

    return torch.stack(embeddings).squeeze(),labels

def neighbor_report(embeddings,labels,model):
    """Prints nearest neighbors in embedding space for each sample.

    Args:
        embeddings (torch.Tensor): the embeddings (nsamples*Nx4L)
        labels (list): the list of length nsamples*N speaker IDs
        model (SpeakerEmbedding object): a model
    """
    N = len(labels)
    for i,label in enumerate(labels):
        dists,idxs = model.dist(embeddings[i].repeat(N,1),embeddings).sort()
        label_idxs = [str(labels[j]) for j in idxs]
        print("%d: %s" % (labels[i],' '.join(label_idxs[:10]))) # hardcoded magic number :(

def visualize_embeddings(embeddings,labels,colors,n):
    """Plots TSNE embeddings of speaker snippet embeddings, labeled by speaker ID.

    Args:
        embeddings (torch.Tensor): the embeddings (nsamples*Nx4L)
        labels (list): the list of length nsamples*N speaker IDs
        colors (ndarray): a Nx3 array of colors; each row is the RGB triplet for one speaker
    """
    # dim reduce
    low_d = TSNE(n_components=2).fit_transform(embeddings.data.cpu().numpy())

    # plot
    plt.clf()
    for i,label in enumerate(labels):
        plt.text(low_d[i,0],low_d[i,1],str(label),color=colors[label,:],fontsize=12)
        plt.scatter(low_d[i,0],low_d[i,1],color=colors[label,:])

    plt.margins(0.1)
    plt.savefig("%d.pdf" % n)

def main(argv):
    """Create model object and train it.
    """
    # parse arguments
    args = parse_all_args()

    data = TinyTimitDataModule(seq_len = args.nframes)

    training_data_sep = data.getTrainSet()
    D = len(training_data_sep)
    #print(len(training_data_sep))
    model = SpeakerEmbedding(D,D,args, training_data_sep)
    trainer = pl.Trainer(max_steps=args.updates, gpus=[0], max_epochs=1, limit_train_batches=.3, limit_val_batches=.1, val_check_interval=0.3, limit_test_batches=0)
    trainer.fit(model,data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main(sys.argv)
