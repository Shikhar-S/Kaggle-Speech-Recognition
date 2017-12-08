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

PATH="./audio"
SAVE_PATH='./Processed'
NOISE="./_background_noise_"
F=open(SAVE_PATH,'w+')



threshold=0.95
threshold_add_noise=0.5
counter=0
Dictionary=["yes","no", "up", "down", "left", "right", "on", "off", "stop", "go","unknown"]
NoisePath=["doing_the_dishes.wav","dude_miaowing.wav","exercise_bike.wav","pink_noise.wav","running_tap.wav","white_noise.wav"]
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

def addNoise(data,dr):
	noise_ratio=random.uniform(0.4,0.8)
	noise_idx=random.randint(0,5)
	noise,r=librosa.load(os.path.join(NOISE,NoisePath[noise_idx]))
	cutfrom=random.randint(0,noise.shape[0]-data.shape[0]-1)
	noise=noise[cutfrom:cutfrom+data.shape[0]]
	assert r==dr
	assert noise.shape[0]==data.shape[0]
	return noise_ratio*noise+(1-noise_ratio)*data

def getFeatures(data,samplerate):
	mfcc=psf.mfcc(data,samplerate,preemph=0)
	mfcc_delta=librosa.feature.delta(mfcc,axis=0)
	data=np.concatenate([mfcc,mfcc_delta],axis=1)
	data=normalize(data)
	return data

def convert_and_save(file,DIR,directory):
    global counter
    FILE=os.path.join(DIR,file)
    data, samplerate=librosa.load(FILE)
    probability=random.uniform(0,1)
    
    data_n=data
    if probability>threshold_add_noise:
    	data_n=addNoise(data,samplerate)

    data=getFeatures(data,samplerate)
    data_n=getFeatures(data_n,samplerate)
    target=np.array(getCode(directory))

    toSave.append((data,data_n,target))
    counter+=1
    if(counter%10==0):
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
        print "Pickling this Directory"
        cPickle.dump(toSave,F)
        toSave=[]
