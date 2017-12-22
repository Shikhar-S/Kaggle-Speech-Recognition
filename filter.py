from keras.layers import LSTM, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import pickle
import numpy as np
import os

maxtimestep=0 
nFeatures=26
BATCH_SIZE=128
nFolders=30 #modify
Epochs=10000

def getData(path):
	#organises data in the format required by neural net
	global maxtimestep
	with open(path) as f:
		data=[]
		for x in range(nFolders):
			data.append(pickle.load(f))
			print "loaded #",x
		for x in data:
			for y in x:
				assert y[0].shape[0]==y[1].shape[0]
				maxtimestep=max(maxtimestep,y[0].shape[0])
		print maxtimestep
		for i,d in enumerate(data):
			for idx,x in enumerate(d):
				noisy=x[0]
				clean=x[1]
				target=x[2]
				if noisy.shape[0]<maxtimestep:
					padwith=maxtimestep-noisy.shape[0]
					noisy=np.pad(noisy,((0,padwith),(0,0)),'constant')
					clean=np.pad(clean,((0,padwith),(0,0)),'constant')
				d[idx]=(noisy,clean)
			g=np.array(d)
			data[i]=g
			print 'done num ',i
		data_np=np.concatenate(data)
		del data
		print data_np.shape
		return (data_np[:,0,:,:],data_np[:,1,:,:])




def Train(x_noisy,x):
	features=Input(shape=(maxtimestep,nFeatures))
	rnn=LSTM(64,return_sequences=True)(features)
	rnn=LSTM(26,return_sequences=True)(rnn)

	Filter=Model(features,rnn)

	callback=[EarlyStopping(monitor='val_loss',patience=5),TensorBoard(histogram_freq=5,batch_size=BATCH_SIZE,write_grads=True)]
	callback.append(ModelCheckpoint(filepath="./Checkpoints",save_best_only=True,period=5))
	
	Filter.compile(optimizer='adadelta',loss='mean_squared_error')
	Filter.fit(x_noisy,x,batch_size=BATCH_SIZE,epochs=Epochs,callbacks=callback,validation_split=0.20,shuffle=True)
	Filter.save('./filter.h5')

x_noisy,x=getData('./Processed')
Train(x_noisy,x)
