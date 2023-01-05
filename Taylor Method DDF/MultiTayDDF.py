#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:48:17 2021

@author: RandallClark
"""
import numpy as np
from numba import njit


class MultiTaylorRC:
    '''
    Initialize the Class
    Parametrs:
        name - The name of the Data set being trained on
        D - The number of dimensions of the data set being trained on and predicted
        T - The highest order of the taylor expansion to train and predict with (i.e. T=2 goes up to X_i*X_j order)
    '''
    def __init__(self,D,T):
        self.D = D
        self.T = T
        self.order = 0
        #Here I create a parameter called order, it essentially counts the total number of terms in the sum (i.e. 1+D+D^2+D^3+...)
        for i in range(self.T+1):
            self.order = self.order + self.D**(i)
    
    """
    #TRAINING STEP------------------------------------------------------------------------------------------------------------
    """
    
    '''
    Train on the Data set using the Taylor Series Expansion method
    This training method utilizes Ridge Regression along with a convenient choice of X and P
    Parameters:
        data - The data set that will be trained on (must be longer than trainlength)
        trainlength - the number of time steps to be trained on
        beta - the size of the regulator term in Ridge Regression
    '''
    def train(self,data,trainlength,beta):
        #We to generate our Y Target from the Data( with dimentions 1 by T where T is total train time), and we need to put together 
        #our X (with dimentions Order by T)
        
        #I've found that nuba, the @njit class decorators, don't like being applied to class functions, so I use a work around method
        #Here I create a numba function, then call it in the next line. We like Numba a LOT because it is a just in timem compiler
        #That compiles all of our slow python code into C++
        funYTarget = self.generateY(data,trainlength,self.D)
        YTarget = funYTarget()
        
        #I Do the same thing as before here, but now generate the X data
        funX = self.generateX(data,trainlength,self.order,self.T,self.D)
        X = funX()
        
        #This should just look like normal Ridge Regression hopefully
        P = np.zeros((self.D,int(self.order)))
        XX = np.linalg.inv(np.matmul(X,np.transpose(X))+beta*np.identity(self.order))
        for i in range(self.D):
            YX = np.matmul(YTarget[i],np.transpose(X))
            P[i] = np.matmul(YX,XX)
        
        
        #Save the parameters from Ridge Regression to be used for Prediction
        self.P = P
        self.beta = beta
        self.endstate = np.zeros(self.D)
        for i in range(self.D):
            self.endstate[i] = data[i][trainlength-1]
        return P
    
    """
    #PREDICTION STEP------------------------------------------------------------------------------------------------------------
    """
    
    '''
    Predict forward using the Data Driven Forcasting Method
    Parameters:
        predictionlength - The amount of time to predict forward
    '''
    def predict(self,predictionlength):
        @njit
        def makeprediction(order,D,endstate,T,P):
            Prediction = np.zeros((predictionlength,D))
            
            #So this should look very similar to what do in the generateX function below, essentially it is trying to put together
            #all possible combinations of 1,x1,..,xD,x1*x1,x1*x2,...,xDxD, and so on for higher order. See the generate X function
            #for it goes into more detail with what is going on here.
            Xpre = np.zeros(order)
            for j in range(T+1):
                displacement = 0
                for l in range(j):
                    displacement = displacement+D**(l)
                for o in range(int(D**j)):
                    placeholder = o
                    Floor = np.zeros(j)
                    for s in range(j):
                        Floor[(j-1-s)] = np.floor(placeholder/(D**(j-1-s)))           
                        placeholder = placeholder - Floor[(j-1-s)]*D**(j-1-s)    
                    
                    Xpre[o+displacement] = 1
                    for r in range(j):
                        Xpre[int(o+displacement)] = Xpre[int(o+displacement)]*endstate[int(Floor[r])] 
            #We've put together our F, now we update
            for d in range(D):
                Prediction[0][d] = endstate[d] + np.dot(P[d],Xpre)
            
            
            #Now we will do this process recursively building the Xpre with the prediction of the last cycles result
            for i in range(1,predictionlength):
                #Construct X vector here
                Xpre = np.zeros(order)
                for j in range(T+1):
                    displacement = 0
                    for l in range(j):
                        displacement = displacement+D**(l)
                    for o in range(int(D**j)):
                        placeholder = o
                        Floor = np.zeros(j)
                        for s in range(j):
                            Floor[(j-1-s)] = np.floor(placeholder/(D**(j-1-s)))           
                            placeholder = placeholder - Floor[(j-1-s)]*D**(j-1-s)         
                        
                        Xpre[o+displacement] = 1
                        for r in range(j):
                            Xpre[int(o+displacement)] = Xpre[int(o+displacement)]*Prediction[i-1][int(Floor[r])]
                
                for d in range(D):
                    #Make predictions
                    Prediction[i][d] = Prediction[i-1][d] + np.dot(P[d],Xpre)
            return Prediction
        
        Prediction = makeprediction(self.order,self.D,self.endstate,self.T,self.P)
        self.Prediction = Prediction
        return Prediction
    
    
    """
    #SECONDARY FUNCTIONS------------------------------------------------------------------------------------------------------------
    """
    
    '''
    This is a tool used to put together the YTarget matrix during training
    '''
    def generateY(self,data,length,D):
        @njit
        def genY():
            #This data is assumed to be D dimensional
            YTarget = np.zeros((D,length))
            for i in range(D):
                for j in range(length):
                    YTarget[i][j] = (data[i][j+1]-data[i][j])
            return YTarget
        return genY
    '''
    This is a tool used to construct the X matrix during training
    '''
    def generateX(self,data,length,order,T,D):
        @njit
        def giveX():
            #This function is overly complicated and the reason why it is so over complicated is because I wanted to create a piece of
            #code that will be able to handle ANY taylor dimension. So this whole lump of code below is written to be able to handle
            #any value for T that the user chooses. Little did I know that the only T's that I ever found useful were T=2 and T=3, which
            #were much easier to build individually. But I have this, so I may as well use it (sometimes it's fun to set T = 4 or 5)

            
            X = np.zeros((order,length))
            
            #We will cycle through the different phases of X (i.e. 1,x,x^2,x^3, and so on)
            for j in range(T+1):
                
                #This displacement counter will count to find the starting point for each cycle
                displacement = 0
                for l in range(j):
                    displacement = displacement+D**(l)
                
                #Cycle though all timepoints
                for i in range(length):
                    
                    #Cycle through all D**j inputs for a cycle
                    for l in range(int(D**j)):
                        #The Floor matrix will count in base D, and its values will span all combinations of X
                        placeholder = l
                        Floor = np.zeros(j)
                        for s in (range(j)):
                            Floor[(j-1-s)] = np.floor(placeholder/(D**(j-1-s)))           #Find out how many D**j can fit in l
                            placeholder = placeholder - Floor[(j-1-s)]*D**(j-1-s)         #Now remove that value*D**j to calculate down one lower order in magnitude
                        
                        #Set the value to 1 (shift the inputs by the amount "displacement")
                        X[l+displacement][i] = 1
                        #Now multiply 1 by all combinations of X, where there are D**j combinations of X for j X's
                        for r in range(j):
                            X[int(l+displacement)][i] = X[int(l+displacement)][i]*data[int(Floor[r])][i]
            return X
        return giveX
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    