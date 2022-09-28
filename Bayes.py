"""
File constaing the Bayes classifier class
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np

class BayesClassifier:
    
    def __init__(self, X1:np.ndarray, X2:np.ndarray, N1:int, N2:int):
        """
        Input:
            X1: np.ndarray, values of class 1
            X2: np.ndarray, values of class 2
            N1: int, length of class 1 array
            N2: int, length of class 2 array
        """
        self._X1 = X1
        self._X2 = X2
        self._N1 = N1
        self._N2 = N2
