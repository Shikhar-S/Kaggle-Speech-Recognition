import numpy as np
from keras.models import load_model
import pickle


MODEL="./filter.h5"
SAVE="./Filtered"
LOAD="./Processed"
maxtimestep=99
nFolder=30


X=[]
Y=[]
def getData():
	with open(LOAD,'rb') as F:
		for _ in range(nFolder):
			data=pickle.load(F)
			for x in data:
				noisy=x[0]
				X.append(noisy)
	return np.array(X)


def Filter():
	X=getData()
	Model=load_model(MODEL)
	Y=Model.predict(batch_size=128)
	saveto=open(SAVE,'wb')
	pickle(Y,saveto)

Filter()

