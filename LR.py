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
            X: np.ndarray shape(N,p-1) p original length of array, array of traningdata
            r: np.ndarray shape(N,), continuous data for response
        
        Output: 
            rhat: np.ndarray shape(N,), best fit data for inputed data response r
        """
        self._X = np.hstack((np.ones((np.shape(X)[0], 1)), X))
        self._r = r

    def __bestmodel(self, X, r):
        """
        Method for calculating the best fit model W
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), r)
    
    def linear_regression(self, X, r):
        """
        Method for calculating the linear regression using best model
        """
        # Using the best fit model to calculate the linear regression 
        try:
            BP = self.__bestmodel(X, r)
            return np.dot(X, BP)
        # Caching errors if method does not work
        except Exception as e:
            print("OPS! Something went wrong. Check {}\n for details.".format(str(e)))
        
    def LeaveOneOut(self):
        """
        Method <>
        """
        
        for i in range(0, np.shape(self._X)[0]):
            _X = self._X[i, :]
            _Rtrain = self._r[i, :]
            



        
        


