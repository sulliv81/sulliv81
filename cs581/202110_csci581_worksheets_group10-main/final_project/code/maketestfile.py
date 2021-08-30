import numpy
import os

f = open("task4tests.txt", "w")
testfolder = '../../../notdata/task4/test'

filelist = os.listdir(testfolder)

for file in filelist:
    print(file)
    f.write('test/' + file + '\n')



f.close()
