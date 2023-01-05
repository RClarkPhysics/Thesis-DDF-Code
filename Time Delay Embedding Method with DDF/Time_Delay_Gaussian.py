#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:34:54 2021

@author: randallclark
"""


import numpy as np
#from numba import njit, prange
from sklearn.cluster import KMeans


class TDEGauss:    
    """
    First Make centers for your Training. It useful to do this step seperately as it can take a while to perform for large data sets,
    and if the use wants to perform multiple trainings to select good hyper parameters, it would be unnecessary to recalculate centers
    every time
    inputs:
        Xdata - Data that will be used for K-means clustering
        P - number of centers you want
    """
    def KmeanCenter(self,Xdata,P,D,length,tau):
        #Data should be input as T by D
        XTau = np.zeros((D,length))
        for d in range(D):
            XTau[D-1-d] = Xdata[tau*d:length+tau*d]
        centers = KMeans(n_clusters=P, random_state=0).fit(XTau.T).cluster_centers_
        return centers
    
    """
    This is how Data Driven Forecasting is performed with a Gaussian function is used as the Radial basis Function for
    interpolatio. Ridge regression is used for training. The Gaussian form is
    e^[(-||X(n)-C(q)||^2)*R]
    inputs:
        Xdata - the data to be used for training and centers
        length - amount of time to train for
        p - used to choose centers. Basic choice: every p data points becomes a center
        beta - regularization term
        R - parameter used in the Radial basis Fucntion
        D - Dimension of th system being trained on
    """
    def FuncApproxF(self,Xdata,length,centers,beta,R,D,tau):
        #We will make our time delay data. This assumes Xdata and stim are 1 dimensional data sets
        XTau = np.zeros((D,length+1))
        for d in range(D):
            XTau[D-1-d] = Xdata[tau*d:length+1+tau*d]
        
        
        #To Create the F(x) we will need only X to generate Y, then give both to the Func Approxer
        #Xdata will need to be 1 point longer than length to make Y
        #Make sure both X and Y are in D by T format
        self.D = D
        self.tau = tau
        XdataT = XTau.T
        Ydata = self.GenY(XTau,length,D)
        NcLength = len(centers)     
        
        #import the centers from the KmeanCenter Functions:
        C = centers
        #Create the Phi matrix with those centers
        PhiMatVolt = self.CreatePhiMatVoltage(XdataT,C,length,NcLength,R,D)
        
        #Perform RidgeRegression
        YPhi = np.matmul(Ydata,PhiMatVolt.T)
        PhiPhi = np.linalg.inv(np.matmul(PhiMatVolt,PhiMatVolt.T)+beta*np.identity(NcLength))
        W = np.matmul(YPhi,PhiPhi)
        self.W = W
            
        #Now we want to put together a function to return
        #@njit
        def newfunc(x):
            
            f = np.matmul(W[0:NcLength],np.exp(-(np.linalg.norm((x-C),axis=1)**2)*R))
            return f
        
        self.FinalX = XTau.T[length-1]
        return newfunc
        
    """
    Predict ahead in time using the F(t)=dx/dt we just built
    input:
        F - This is the function created above, simply take the above functions output and put it into this input
        PreLength - choose how long you want to predict for
        Xstart - Choose where to start the prediction, the standard is to pick when the training period ends, but you can choose it
                    to be anywhere.
    """
    def PredictIntoTheFuture(self,F,PreLength,PData):
        #stim must be 1 value longer than PreLength
        #An Important Detail regarding PData. Your Prediction will start at the tau*(D-1) data point, everything before that
        #must be from the training phase, or other if you are driving, this will be necessary to update the delayed vectors
        #tau*(D-1) is the last point from the training data
        
        #@njit
        def makePre(D,PData,tau):           
            Prediction = np.zeros(PreLength+tau*(D-1)+1)
            Prediction[0:tau*(D-1)+1] = PData[0:tau*(D-1)+1]
            
            #Start by Forming the Bases
            Input = np.flip(Prediction[0:tau*D:tau])
            Prediction[tau*(D-1)+1] = Prediction[tau*(D-1)]+F(Input)
            
            #Let it run forever now
            for t in range(1,PreLength):
                Input = np.flip(Prediction[t:t+tau*D:tau])
                Prediction[t+tau*(D-1)+1] = Prediction[t+tau*(D-1)]+F(Input)
            return Prediction
        
        Prediction = makePre(self.D,PData,self.tau)
        return Prediction.T
    
    """
    These are secondary Functions used in the top function. You need not pay attention to these unless you wish to understand or alter
    the code.
    """
    def CreatePhiMatVoltage(self,X,C,Nlength,NcLength,R,D):
        #@njit
        def getMat():
            Mat = np.zeros((NcLength,Nlength),dtype = 'float64')
            for i in range(NcLength):
                CC = np.zeros((Nlength,D))
                CC[:] =  C[i]
                Diff = X[0:Nlength]-CC
                Norm = np.linalg.norm(Diff,axis=1)
                Mat[i] = Norm
            Mat[0:NcLength][0:Nlength] = np.exp(-(Mat[0:NcLength][0:Nlength]**2)*R)
            """
            Mat = np.zeros((NcLength+1,Nlength),dtype = 'float64')
            XX = np.zeros((NcLength,Nlength,D),dtype = 'float64')
            CC = np.zeros((Nlength,NcLength,D),dtype = 'float64')
            XX[0:Nlength][0:Nlength] = X[0:Nlength].astype('float64')
            CC[0:Nlength][0:Nlength] = C.astype('float64')

            Diff = XX-np.swapaxes(CC,0,1)
            NormXC = np.linalg.norm(Diff,axis=2)
            Mat[0:NcLength][0:Nlength] = np.exp(-(NormXC**2)*R)
            Mat[NcLength] = 0.5*(stim[0:Nlength].astype('float64')+stim[1:1+Nlength].astype('float64'))
            """
            return Mat
        Mat = getMat()
        return Mat

    def GenY(self,Xdata,length,D):
        #@njit
        def makeY():
            Y = Xdata[0][1:length+1]-Xdata[0][0:length]
            
            #Y = np.zeros(length)
            #for t in range(length):
            #    Y[t] = Xdata[0][t+1]-Xdata[0][t]
            return Y
        Y = makeY()
        return Y