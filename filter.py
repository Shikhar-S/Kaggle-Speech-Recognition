from keras.layers import LSTM
from keras.models import Model
import pickle
import numpy as np
import os

maxtimestep=0 #fill this
nFeatures=26
BATCH_SIZE=128
nFolders=0 #modify

def getData(path):
	global maxtimestep
	with open(path) as f:
		data=[]
		for x in range(nFolders):
			data.append(pickle.load(f))
		for x in data:
			assert x[0].shape[0]==x[1].shape[0]
			maxtimestep=max(maxtimestep,x[0].shape[0])
		
		for idx,x in enumerate(data):
			noisy=x[0]
			clean=x[1]
			target=x[2]
			if noisy.shape[0]<maxtimestep:
				padwith=maxtimestep-noisy.shape[0]
				noisy=np.pad(noisy,((0,padwith),(0,0)),'constant')
				clean=np.pad(clean,((0,padwith),(0,0)),'constant')
			data[idx]=(noisy,clean)
		return (data[:][0],data[:][1])




def Train(x_noisy,x):
	features=Input(shape=(maxtimestep,nFeatures))
	rnn=LSTM(64,return_sequences=True)(features)
	rnn=LSTM(26,activation=None,return_sequences=True)(rnn)

	Filter=Model(features,rnn)

	Filter.compile(optimizer='adadelta',loss='mean_squared_error')
	Filter.fit(x_noisy,x,batch_size=BATCH_SIZE,validation_split=0.20,shuffle=True)
	Filter.save('./filter.h5')


x_noisy,x=getData('./Processed')
Train(x_noisy,x)
