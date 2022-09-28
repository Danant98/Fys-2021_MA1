"""
File constaing the Bayes classifier class
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np
from math import gamma

class BayesClassifier:
    
    def __init__(self, X1:np.ndarray, X2:np.ndarray, alpha:float=9):
        """
        Input:
            X1: np.ndarray, values of class 1
            X2: np.ndarray, values of class 2
            alpha: float, 
        """
        self._X1 = X1
        self._X2 = X2
        self._alpha = alpha 

    def beta_hat(self):
        """
        Method for calculating the estimator for beta
        """
        return (1 / (self._alpha* len(self._X1))) * np.sum(self._X1)

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

    def Gamma(self, X:np.ndarray):
        """
        Method for calculating the gamma distribution

        Input:
            X: np.ndarray, array constaing the x values
        """
        return (1 / (self.beta_hat()**self._alpha * gamma(self._alpha))) * X**(self._alpha - 1) * np.exp(-X / self.beta_hat())

    def Normal(self, X:np.ndarray):
        """
        Method for calculating the normal distribution 

        Input: 
            X: np.ndarray, array constaing the x values
        """
        return (1 / (np.sqrt(2 * np.pi) * self.variance())) * np.exp(-(np.square(self._X2 - self.mean()) / 2 * np.square(self.variance())))







