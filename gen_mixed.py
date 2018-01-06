# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:00:46 2017

@author: Shikhar
"""
import pickle
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
F=open(SAVE_PATH,'wb')


TIMESTEP=100 #number of frames per audio (assuming 10 millisecond frame, 1 second audio)
FEATURES=26 #number of features per frame
threshold=0.95 #to get equal mix of unknown target type.
threshold_add_noise=0.5 #half of the files are mixed with noise others copied as is.
counter=0 
Dictionary=["yes","no", "up", "down", "left", "right", "on", "off", "stop", "go","unknown"] #targets
NoisePath=["doing_the_dishes.wav","dude_miaowing.wav","exercise_bike.wav","pink_noise.wav","running_tap.wav","white_noise.wav"] #noise files
RevDict={}
for i,x in enumerate(Dictionary):
    RevDict[x]=i

toSave=[]
def getCode(x):
    try:
        return RevDict[x]
    except:
        return RevDict['unknown'] #exception for non targets


def isTargetDir(x):
    if(x in Dictionary):
        return True
    else:
        return False

def addNoise(data,dr):
    #mixes noise in data at sample rate dr 
	noise_ratio=random.uniform(0.4,0.8)
	noise_idx=random.randint(0,5)
	noise,r=librosa.load(os.path.join(NOISE,NoisePath[noise_idx]))
	cutfrom=random.randint(0,noise.shape[0]-data.shape[0]-1)
	noise=noise[cutfrom:cutfrom+data.shape[0]] #generate an equal length noise clip from the longer recording
	assert r==dr
	assert noise.shape[0]==data.shape[0]
	return noise_ratio*noise+(1-noise_ratio)*data

def getFeatures(data,samplerate):
    #generates features from data, namely 13 mfcc coefficients and their derivatives
	mfcc=psf.mfcc(data,samplerate,preemph=0)
	mfcc_delta=librosa.feature.delta(mfcc,axis=0)
	data=np.concatenate([mfcc,mfcc_delta],axis=1)
	data=normalize(data) #not sure if this is the right thing to do.
    #TO-DO : make time step equal for all files.
    if data.shape[0]<TIMESTEP:
        required=TIMESTEP-data.shap[0]
        req_array=np.zeros(shape=(required,FEATURES))
        data=np.concatenate([data,req_array],axis=0)
    elif data.shape[0]>TIMESTEP:
        data=data[:TIMESTEP,:]
	return data #size- time*26

def convert_and_save(file,DIR,directory):
    #preprocesses an audio file-> 'file' to get features by calling above utility fns
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

num_pickels=0
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
        pickle.dump(toSave,F) #pickles directory after directory one after other. Has to be retrieved multiple times when reading. Google pickling multiple files.
        num_pickels+=1
        toSave=[]
F.close()
print(num_pickels, "pickle must be reloaded")

