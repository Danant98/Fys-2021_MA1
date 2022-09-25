"""
File containg the multivaritate linear regression class
"""
# Import libraries and modules
import numpy as np


class LinearRegression:

    def __init__(self, X:np.ndarray, r:np.ndarray):
        """
        Input:
            X: np.ndarray, Inputed datamatrix for traningdata 
            r: np.ndarray, Inputed continuous data for response
        
        Output: np.ndarray, best fit data for inputed data response r
        """
        
        self._X = np.hstack((np.ones((np.shape(X)[0], 1)), X))
        self._r = r

    def __bestfit(self):
        """
        Method for calculating the best fit for the linear regression of r 
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(self._X.T, self._X)), self._X.T), self._r)
    
    def linear_regression(self):
        """
        Method for runing the linear regression
        """
        self.__bestfit()


