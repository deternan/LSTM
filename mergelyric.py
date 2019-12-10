# coding=utf8
'''
Merge all lyric files
version: December 09, 2019 04:13 PM
Last revision: December 10, 2019 09:32 AM

Author : Chao-Hsuan Ke
'''

import os
from os import listdir
from os.path import isfile, isdir, join

filePath = "D:\\Phelps\\GitHub\\Python\\LSTM\\data\\MayDay\\";        # folder name
files = listdir(filePath)

# ouptut
outputFolder = "D:\\Phelps\\GitHub\\Python\\LSTM\\data\\"                      # output folder name
outputFile = "alllyric_2.txt"
fw = open(outputFolder + outputFile, "a", encoding='UTF-8')       # 'a' --> overlapping

contentlist = []

def readfilecontent(filepath):
    f = open(filepath, 'r', encoding="utf-8")
    data = f.read()
#    lines = data.split('\\s')
#    contentlist.append(lines)
    contentlist.append(data)


def readfolderfileName(foldername):
    folderfilepath = listdir(foldername)
    for fileNames in folderfilepath:
        # print(fileNames)
        readfilecontent(foldername + fileNames);

if (os.path.exists(filePath)):
    for fileName in files:
        fullpath = join(filePath, fileName)
        if isfile(fullpath):
            readfilecontent(fullpath);
            #print("file：", fileName)
        elif isdir(fullpath):
            #print("folder：", fullpath)
            readfolderfileName(fullpath + '\\')

# output (1), split space
# for ss in contentlist:
#     tmpStr = ss[0].replace('\u3000', ' ')
#     tmpStr = tmpStr.replace(' ', ' ')
#     tmplist = tmpStr.split(' ')
#     print(len(tmplist), tmplist[0], tmplist)
#     #print(len(tmplist), ss[0])
#     for txtindex in tmplist:
#         if(len(txtindex.strip()) > 0):
#             fw.write(txtindex.strip()+ '\n')
#             print(txtindex.strip())

# output (2), split space
for ss in contentlist:
    if(len(ss.strip()) > 0):
        fw.write(ss+ '\n');

print('finished')
fw.close()
