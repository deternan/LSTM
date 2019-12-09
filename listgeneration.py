# coding=utf8
'''
version: December 09, 2019 02:01 PM
Last revision: December 09, 2019 02:40 PM

Author : Chao-Hsuan Ke
'''

import os
from os import listdir
from os.path import isfile, isdir, join


filePath = "D:\\Phelps\\GitHub\\Python\\LSTM\\data\\MayDay\\";        # folder name
files = listdir(filePath)
list = []

# ouptut
outputFolder = "D:\\Phelps\\GitHub\\Python\\LSTM\\data\\"                      # output folder name
outputFile = "list.txt"
fw = open(outputFolder + outputFile, "a", encoding='UTF-8')       # 'a' --> overlapping

def readfolderfileName(foldername):
    folderfilepath = listdir(foldername)
    for fileNames in folderfilepath:
        #print(fileNames[0:len(fileNames)-4])
        # print(fileNames)
        list.append(fileNames[0:len(fileNames)-4])



if (os.path.exists(filePath)):
    for fileName in files:
        fullpath = join(filePath, fileName)
        if isfile(fullpath):
            list.append(fileName[0:len(fileName)-4])
            #print("file：", fileName)
        elif isdir(fullpath):
            #print("folder：", fileName)
            readfolderfileName(fullpath)

for ss in list:
    fw.write(ss+"\n")
    print(ss)

fw.close()

print('files list: ',list)