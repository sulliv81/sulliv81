import os
import numpy as np
import random




class Speaker():
    def __init__(self, speakerid):
        self.speakerid = speakerid
        self.filelist = []

    def add(self, filename):
        self.filelist.append(filename)

    def getlist(self):
        return self.filelist

    def getid(self):
        return self.speakerid

def printspklist(speakers):
    for speak in speakers:
        print("SpeakerID : ", speak.getid())
        for file in speak.getlist():
            print("Has File: ", file)

fname = '../notdata/train'
datafolder = os.listdir(fname)
speakerlist = []


datafolder.sort()

for data in datafolder:
    check = data.split('.')[1]
    if check ==  'wav' :
        num = int(data.split('.')[0].split("file")[1])
        wavfile = data
        print('wavfile: |', data, '|')
        spkpath = fname + '/file' + str(num) + '.spkid.txt'
        spkid = int(np.loadtxt(spkpath))
        print('Filenumber: ', num, "SpkID: ", spkid)
        found = False
        for spk in speakerlist:
            if spk.getid() == spkid:
                spk.add(wavfile)
                found = True
        if (found == False):
            newspeaker = Speaker(spkid)
            newspeaker.add(wavfile)
            speakerlist.append(newspeaker)

f = open("trainspeakerpairs.txt", "x")
printspklist(speakerlist)
speakerlength = len(speakerlist)
for i in range (500):
    if (random.random() > .5):
        #make same pair
        print("make same")
        spkidx = random.randint(0, speakerlength-1)
        speaker = speakerlist[spkidx]
        spklist = speaker.getlist()
        numfiles = len(spklist)
        if (numfiles > 1):
            file1idx = random.randint(0, numfiles-1)
            file1 =  spklist[file1idx]
            file2idx = random.randint(0, numfiles-1)
            while(file2idx == file1idx):
                file2idx = random.randint(0, numfiles-1)

            file2 = spklist[file2idx]
            f.write('train/' + file1 + ' ' + 'train/' +  file2 + '\n')


    else:
        #make diff pair
        print("make diff")
        spk1idx = random.randint(0, speakerlength-1)
        speaker1 = speakerlist[spk1idx]
        spk2idx = random.randint(0, speakerlength-1)
        while(spk2idx == spk1idx):
            spk2idx = random.randint(0, speakerlength-1)

        speaker2 = speakerlist[spk2idx]

        spk1list = speaker1.getlist()
        spk2list = speaker2.getlist()
        spk1listlen = len(spk1list)
        spk2listlen = len(spk2list)

        spk1listidx = random.randint(0, spk1listlen-1)
        spk2listidx = random.randint(0, spk2listlen-1)
        file1 = spk1list[spk1listidx]
        file2 = spk2list[spk2listidx]
        f.write('train/' + file1 + ' ' + 'train/' + file2 + '\n')



f.close()
