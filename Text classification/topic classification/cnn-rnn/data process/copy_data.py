# coding=utf8

'''
version: December 19 2019, 05:31 PM
Last revision: December 20 2019, 02:53 PM

Author : Chao-Hsuan Ke
'''

import os
import random
from os import listdir
from os.path import isfile, isdir, join
import shutil

filePath = 'D:\\data model\\THUCTC\\THUCNews\\';  # folder name
targetPath = 'D:\\data model\\THUCTC\\THUCNews_tinydata\\'

def readFile(filePath):
    files = listdir(filePath)
    for fileName in files:
        fullpath = join(filePath, fileName)
        fileCount(fullpath, fileName)

fileList = []

def copyFiles(sourcepath, obj_path, fileindex):
    files = listdir(sourcepath)
    count = 0
    maxNameTmp = max(files)
    maxName = maxNameTmp.split('.')
    minNameTmp = min(files)
    minName = minNameTmp.split('.')
    for fileName in fileindex:
        fileName = fileindex[count]
        filenameint = int(fileName) + int(minName[0])
        filenamestr = str(filenameint)
        #print(filenamestr)
        copyName = sourcepath + '\\' + str(filenamestr) + '.txt'
        targetName = obj_path + '\\' + str(filenamestr) + '.txt'
        count+=1
        shutil.copyfile(copyName, targetName)
    print(obj_path, 'finished')


def randomNumber(total, sourcepath, targetPath):
    li = [i for i in range(total)]
    res = []
    num = 6500
    for i in range(num):
        t = random.randint(i, total - 1)
        res.append(li[t])
        li[t], li[i] = li[i], li[t]
    #print(res)
    # copy files index to new target folder
    copyFiles(sourcepath, targetPath, res)


def fileCount(folderpath, targetFolder):
    for root, subFolders, files in os.walk(folderpath):
        for file in files:
            f = os.path.join(root, file)
            fileList.append(f)
        # generate random list
        randomNumber(len(fileList), folderpath, targetPath+targetFolder)
        fileList.clear()



# read folder and count the files
readFile(filePath)
