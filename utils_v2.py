import cPickle
import os
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split



def data_lists_to_batches(inputList, targetList, batchSize):
    '''Takes a list of input matrices and a list of target arrays and returns
       a list of batches, with each batch being a 3-element tuple of inputs,
       targets, and sequence lengths.
       inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
       targetList: list of 1-d arrays or lists of ints
       batchSize: int indicating number of inputs/targets per batch
       returns: dataBatches: list of batch data tuples, where each batch tuple (inputs, targets, seqLengths) consists of
                    inputs = 3-d array w/ shape nTimeSteps x batchSize x nFeatures
                    targets = tuple required as input for SparseTensor
                    seqLengths = 1-d array with int number of timesteps for each sample in batch
                maxSteps: maximum number of time steps across all samples'''
    print(len(inputList),len(targetList))
    assert len(inputList) == len(targetList)
    nFeatures = inputList[0].shape[0]
    maxSteps = 0
    for inp in inputList:
        maxSteps = max(maxSteps, inp.shape[1])
    print(maxSteps)
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []
    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((maxSteps, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            padSecs = maxSteps - inputList[origI].shape[1]             
            z= np.pad(inputList[origI].T, ((0,padSecs),(0,0)),'constant')
            batchInputs[:,batchI,:]=z
            batchTargetList.append(targetList[origI])
        batchTargetList=np.array(batchTargetList)
        batchedTargetList=np.squeeze(batchTargetList)
        dataBatches.append((batchInputs, batchTargetList,batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxSteps,len(inputList))

def load_batched_data(specPath, batchSize):
    '''returns 3-element tuple: batched data (list), max # of time steps (int), and
       total number of samples (int)'''
    with open(specPath) as FILE:
        data=cPickle.load(FILE)
        print('pickle loaded')
        X=[x[0].T for x in data]
        Y=[y[1] for y in data]
        print('Data Segmented')
        del data
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
        z=data_lists_to_batches(X_train,Y_train,batchSize) + data_lists_to_batches(X_test,Y_test,batchSize)
        return z





