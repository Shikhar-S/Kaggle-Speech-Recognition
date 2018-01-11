# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:58:56 2018

@author: t-shbhar
"""
#Model using LSTM 

import pickle
from random import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras.layers import Input, LSTM
from keras.models import Model

SAVED_PATH='./Processed'
num_pickles=30
data=[]
target=[]
TIMESTEP=100 #number of frames per audio (assuming 10 millisecond frame, 1 second audio)
FEATURES=26 #number of features per frame
BATCH_SIZE=128
first_sz=64
second_sz=32
third_sz=26
checkpoint="./Checkpoint_filter_v1"

def get_data(i,x):
    F=open(SAVED_PATH,"rb")
    for _ in range(num_pickles):
        L=pickle.load(F)                    #L is a list of len (num_files* 3) where first two have shape (100*26) and third has (1,)
        for audio in L:
            x.append(audio[i])
    F.close()

def build_model():
    inputs=Input(shape=(TIMESTEP,FEATURES))
    first=LSTM(first_sz,return_sequences=True)(inputs)
    second=LSTM(second_sz,return_sequences=True)(first)
    predictions=LSTM(third_sz,return_sequences=True)(second)
    model=Model(inputs=inputs,outputs=predictions)
    model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
    return model

#prepare data
get_data(1,data) #0 for data without noise ,1 for with noise,2 for target
get_data(0,target)
data=np.array(data)
data=np.squeeze(data)
target=np.squeeze(np.array(target))

#shuffle data
number_of_audio=data.shape[0]
index_list=[i for i in range(number_of_audio)]
shuffle(index_list)
data=data[index_list,:,:]
target=target[index_list,:,:]


#build model
model=build_model()
callbacks=[]
callbacks.append(ModelCheckpoint(checkpoint,monitor='val_loss',save_best_only=True))
callbacks.append(EarlyStopping(patience=3))
model.fit(data,target,batch_size=BATCH_SIZE,epochs=1000,callbacks=callbacks,validation_split=0.2,shuffle=True)
model.save('filter_v1.h5')
del model



