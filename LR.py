#!/usr/bin/env python
"""
File containg the multivaritate linear regression class
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np

class LinearRegression:

    def __init__(self, X:np.ndarray, r:np.ndarray):
        """
        Input:
            X: np.ndarray shape(N,p-1) p original length of array, array of data
            r: np.ndarray shape(N,), continuous data for response
        """
        self._X = np.hstack((np.ones((np.shape(X)[0], 1)), X))
        self._r = r

    def __bestmodel(self, X:np.ndarray, r:np.ndarray):
        """
        Method for calculating the best fit model W
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), r)
    
    def __linear_regression(self, BM:np.ndarray, X:np.ndarray):
        """
        Method for calculating the linear regression using best model.

        Input: 
            X: np.ndarray, array of data
            BM: np.ndarray, array containg best model for constants in linear regression
            r: np.ndarray, continuous data for response
        """
        # Using the best fit model to calculate the linear regression 
        try:
            return np.dot(X, BM)
        # Caching errors if method does not work
        except Exception as e:
            print("OPS! Something went wrong. Check {0:}\n for details.".format(str(e)))
        
    def LeaveOneOut(self):
        """
        Method for implementing leave-one-out cross-validation. 

        Output: 
            rhat: np.ndarray (N,), returning array containg 
        """
        # Holding the values for validation data
        _rhat = np.zeros(np.shape(self._r))
        residuals = np.zeros(np.shape(self._r))
        for i in range(0, np.shape(self._X)[0]):
            Xtest = self._X[i, :]
            Rtest = self._r[i]
            Xtrain = np.delete(self._X, i, 0)
            Rtrain = np.delete(self._r, i, 0)

            BM = self.__bestmodel(Xtrain, Rtrain)
            _rhat[i] = self.__linear_regression(BM, Xtest)
            residuals[i] = Rtest - _rhat[i]
        
        return _rhat, np.abs(residuals)

