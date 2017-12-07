# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:00:46 2017

@author: Shikhar
"""
import cPickle
import librosa
import os
import random
import numpy as np
import sys
from sklearn.preprocessing import normalize
import python_speech_features as psf

PATH="./train/audio"
SAVE_PATH='./Processed'
F=open(SAVE_PATH,'w+')

threshold=0.95
counter=0
Dictionary=["yes","no", "up", "down", "left", "right", "on", "off", "stop", "go","unknown"]
RevDict={}
for i,x in enumerate(Dictionary):
    RevDict[x]=i

toSave=[]
def getCode(x):
    try:
        return RevDict[x]
    except:
        return RevDict['unknown']


def isTargetDir(x):
    if(x in Dictionary):
        return True
    else:
        return False

def convert_and_save(file,DIR,directory):
    global counter
    FILE=os.path.join(DIR,file)
    data, samplerate=librosa.load(FILE)
    mfcc=psf.mfcc(data,samplerate,preemph=0)
    mfcc_delta=librosa.feature.delta(mfcc,axis=0)
    target=np.array(getCode(directory))
    data=np.concatenate([mfcc,mfcc_delta],axis=1)
    data=normalize(data)
    toSave.append((data,target))
    counter+=1
    if(counter%100==0):
        print counter," -- ", 
        sys.stdout.flush()


for directory in os.listdir(PATH):
    DIR=os.path.join(PATH,directory)
    if(os.path.isdir(DIR)):
        print 'working with', directory,"--> "
        if(not isTargetDir(directory)):
            for fl in os.listdir(DIR):
                probability=random.uniform(0,1)
                if(probability>threshold):
                   convert_and_save(fl,DIR,directory) 
        else:
            for fl in os.listdir(DIR):
                convert_and_save(fl,DIR,directory)    
        print "Done with ", directory

print 'pickling'
cPickle.dump(toSave,F)
