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

    def beta_hat(self, X:np.ndarray):
        """
        Method for calculating the estimator for beta

        Input: 
            X: np.ndarray, array constaining 

        """
        return (1 / (self._alpha* len(X))) * np.sum(X)

    def mean(self, X:np.ndarray):
        """
        Method for calculating the estimator for the mean
        """
        return (1 / len(X)) * np.sum(X)

    def variance(self, X:np.ndarray):
        """
        Method for calculating the estimator for the variance 
        """
        return (1 / len(X)) * np.sum(np.square(X - self.mean(X)))

    def Gamma(self, X:np.ndarray):
        """
        Method for calculating the gamma distribution

        Input:
            X: np.ndarray, array constaing the x values
        """
        return (1 / (self.beta_hat(X)**self._alpha * gamma(self._alpha))) * X**(self._alpha - 1) * np.exp(-X / self.beta_hat(X))

    def Normal(self, X:np.ndarray):
        """
        Method for calculating the normal distribution 

        Input: 
            X: np.ndarray, array constaing the x values
        """
        return (1 / (np.sqrt(2 * np.pi) * self.variance(X))) * np.exp(-(np.square(X - self.mean(X)) / 2 * np.square(self.variance(X))))







