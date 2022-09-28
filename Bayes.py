"""
File constaing the Bayes classifier class
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np

class BayesClassifier:
    
    def __init__(self, X1:np.ndarray, X2:np.ndarray):
        """
        Input:
            X1: np.ndarray, values of class 1
            X2: np.ndarray, values of class 2
            N1: int, length of class 1 array
            N2: int, length of class 2 array
        """
        self._X1 = X1
        self._X2 = X2

    def beta_hat(self, alpha:float=9):
        """
        Method for calculating the estimator for beta
        """
        return (1 / (alpha * len(self._X1))) * np.sum(self._X1)

    def mean(self):
        """
        Method for calculating the estimator for the mean
        """
        return (1 / len(self._X2)) * np.sum(self._X2)

    def variance(self):
        """
        Method for calculating the estimator for the variance 
        """
        return (1 / len(self._X2)) * np.sum(np.square(self._X2 - self.mean()))







