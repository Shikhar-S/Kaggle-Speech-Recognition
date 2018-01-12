# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:10:38 2018

@author: t-shbhar
"""

import numpy as np
from keras.models import load_model
import pickle
import pandas as pd
import os
import librosa
import python_speech_features as psf
from sklearn.preprocessing import normalize

MODEL="./audio_classifier_v1.h5"
FILTER="./filter_v1.h5"
TEST_DATA="./test/audio"
TIMESTEP=100 #number of frames per audio (assuming 10 millisecond frame, 1 second audio)
FEATURES=26 #number of features per frame
BATCH_SIZE=128
inputs=[]
outputs=[]

Dictionary=["yes","no", "up", "down", "left", "right", "on", "off", "stop", "go","unknown"] #targets



c=0
for file in os.listdir(TEST_DATA):
    c+=1
    data,samplerate=librosa.load(os.path.join(TEST_DATA,file))
    mfcc=psf.mfcc(data,samplerate,preemph=0,nfft=1024)
    mfcc_delta=librosa.feature.delta(mfcc,axis=0)
    data=np.concatenate([mfcc,mfcc_delta],axis=1)
    data=normalize(data)
    if data.shape[0]<TIMESTEP:
        required=TIMESTEP-data.shape[0]
        req_array=np.zeros(shape=(required,FEATURES))
        data=np.concatenate([data,req_array],axis=0)
    elif data.shape[0]>TIMESTEP:
        data=data[:TIMESTEP,:]
    inputs.append(data)
    if(c%100==0):
        print(c,"--",end=' ')
     

inputs=np.array(inputs)
print(inputs.shape)


Filter=load_model(FILTER)
print('Filter Loaded')
Model=load_model(MODEL)
print('Model loaded')
y=Filter.predict(inputs,batch_size=BATCH_SIZE)
print('Filtered all files')
y=Model.predict(y,batch_size=BATCH_SIZE)
print('Predicted all files')
y=np.argmax(y,axis=1)
i=0
c=0
for file in os.listdir(TEST_DATA):
    outputs.append([file,Dictionary[y[i]]])
    i+=1
    if(i%100==0):
        print(i,"--c-",end=' ')
       


df=pd.DataFrame(outputs)
df.to_csv('upload_noisy.csv',index=False,header=['fname','label'])



