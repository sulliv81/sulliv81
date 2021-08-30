import os
import numpy as np


fname = 'notdata'



datafolder = os.listdir(fname)
outlist = []
datafolder.sort()
f = open("trainingdata.txt", "x")
for data in datafolder:
    check = data.split('.')[1]
    if check ==  'wav' :
        num = data.split('.')[0].split("file")[1]
        print(num)
        f.write(str(num) + '\n')

f.close()
