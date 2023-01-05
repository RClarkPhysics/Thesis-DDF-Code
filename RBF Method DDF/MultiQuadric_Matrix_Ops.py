#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:34:54 2021

@author: randallclark
"""


import numpy as np
from sklearn.cluster import KMeans


class MultiQMat:    
    """
    First Make centers for your Training. It useful to do this step seperately as it can take a while to perform for large data sets,
    and if the user wants to perform multiple trainings to select good hyper parameters, it would be unnecessary to recalculate centers
    every time
    inputs:
        Xdata - 1 Dimensional Data that will be used for K-means clustering.
        P - number of centers you want
        D - The number of dimensions you want
        tau - the time delay you want
    """
    def KmeanCenter(self,Xdata,P,length):
        centers = KMeans(n_clusters=P, random_state=0).fit(Xdata.T[:length]).cluster_centers_
        return centers
    
    """
    This is how Data Driven Forecasting is performed with a Gaussian function is used as the Radial basis Function for
    interpolatio. Ridge regression is used for training. The MultiQuadric form is
    [(||X(n)-C(q)||^2) + R]^alpha
    inputs:
        Xdata - the data to be used for training and centers
        length - amount of time to train for
        C - My Centers
        beta - regularization term
        R - parameter used in the Radial basis Fucntion
        alpha - the exponent parameter in the Radial Basis Function
        D - Dimension of th system being trained on
    """
    def FuncApproxF(self,Xdata,length,C,beta,R,alpha,D):
        #To Create the F(x) we will need only X to generate Y, then give both to the Func Approxer
        #Xdata will need to be 1 point longer than length to make Y
        #Make sure both X and Y are in D by T format
        self.D = D
        XdataT = Xdata.T
        Ydata = self.GenY(Xdata,length,D)
        NcLength = len(C)
        
        #Creat the Phi matrix with those centers
        PhiMat = self.CreatePhiMat(XdataT,C,length,NcLength,R,alpha,D)
        
        #Perform RidgeRegression
        YPhi = np.zeros((D,NcLength))
        for d in range(D):
            YPhi[d] = np.matmul(Ydata[d],PhiMat.T)
        PhiPhi = np.linalg.inv(np.matmul(PhiMat,PhiMat.T)+beta*np.identity(NcLength))
        W = np.zeros((D,NcLength)) 
        for d in range(D):
            W[d] = np.matmul(YPhi[d],PhiPhi)
            
        #Now we want to put together a function to return
        def newfunc(x):
            f = np.matmul(W,((np.linalg.norm((x-C),axis=1)**2) + R)**alpha)
            return f
        
        self.FinalX = Xdata.T[length-1]
        self.W = W
        self.PhiMat = PhiMat
        return newfunc
        
    """
    Predict ahead in time using the F(t)=dx/dt we just built
    input:
        F - This is the function created above, simply take the above functions output and put it into this input
        PreLength - choose how long you want to predict for
        Xstart - Choose where to start the prediction, the standard is to pick when the training period ends, but you can choose it
                    to be anywhere.
    """
    def PredictIntoTheFuture(self,F,PreLength,Xstart):
        def makePre(D,FinalX):
            Prediction = np.zeros((PreLength,D))
            #Start by Forming the Bases
            Prediction[0] = FinalX+F(FinalX)
            
            #Let it run forever now
            for t in range(1,PreLength):
                Prediction[t] = Prediction[t-1]+F(Prediction[t-1])
            return Prediction
        
        Prediction = makePre(self.D,Xstart)
        return Prediction.T
    
    """
    These are secondary Functions used in the top function. You need not pay attention to these unless you wish to understand or alter
    the code.
    """     
    
    def CreatePhiMat(self,X,C,Nlength,NcLength,R,alpha,D):
        Mat = np.zeros((NcLength,Nlength),dtype = 'float64')
        for i in range(NcLength):
            CC = np.zeros((Nlength,D))
            CC[:] =  C[i]
            Diff = X[0:Nlength]-CC
            Norm = np.linalg.norm(Diff,axis=1)
            Mat[i] = Norm
        Mat[0:NcLength][0:Nlength] = ((Mat[0:NcLength][0:Nlength]**2) + R)**alpha
        return Mat

    def GenY(self,Xdata,length,D):
        #This code is self explanatory. Take the difference
        Y = np.zeros((D,length))
        for d in range(D):
            Y[d] = Xdata[d][1:length+1]-Xdata[d][0:length]
        return Y
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    