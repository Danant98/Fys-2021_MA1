"""
File constaing the Bayes classifier class
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np
from math import gamma, pow

class BayesClassifier:
    
    def __init__(self, X1:np.ndarray, X2:np.ndarray, labels:np.ndarray, alpha:float=9):
        """
        Input:
            X1: np.ndarray, values of class 1 from trainingset
            X2: np.ndarray, values of class 2 from trainingset
            labels: np.ndarray, labels for total amount of classes from traningset
            alpha: float, constant used in gamma function and gamma distribution
        """
        self._X1 = X1
        self._X2 = X2
        self._labels = np.unique(labels)
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

        Output:
            np.ndarray, 
        """
        return (1 / len(self._X2)) * np.sum(np.square(self._X2 - self.mean()))

    def Gamma(self, X:np.ndarray):
        """
        Method for calculating the gamma distribution

        Input:
            X: np.ndarray, array containg the x values

        Output:
            np.ndarray, array containg the values of the gamma distribution probability function
        """
        return (1 / (self.beta_hat()**(self._alpha) * gamma(self._alpha))) * (X**(self._alpha - 1)) * np.exp(-X / self.beta_hat())

    def Normal(self, X:np.ndarray):
        """
        Method for calculating the normal distribution 

        Input: 
            X: np.ndarray, array containg the x values
        
        Output:
            np.ndarray, array containg the values of the normal distribution probability function
        """
        return (1 / (np.sqrt(2 * np.pi) * np.sqrt(self.variance()))) * np.exp(-(np.square(X - self.mean()) / (2 * self.variance())))

    def prior_prob(self):
        """
        Method for calculating the prior probabilities

        Output:
        """
        PC0 = len(self._X1) / len(self._labels)
        PC1 = len(self._X2) / len(self._labels)
        return PC0, PC1


    def PredictedY(self, testX:np.ndarray):
        """
        
        """
        predictions = np.array([self._prediction(x) for x in testX])
        return predictions

    def _prediction(self, testx:float):
        """
        Method for calculating the predictions

        Input:
            testx: float, inputed x value from testset 
        """
        # Calculating
        posterior = np.zeros(len(self._labels))
        for num in range(0, len(self._labels)-1):
            if int(self._labels[num]) == 0:
                posterior[num] = np.sum(np.log(self.Gamma(testx))) + np.log(self.prior_prob()[0])
            elif int(self._labels[num]) == 1:
                posterior[num] = np.sum(np.log(self.Normal(testx))) + np.log(self.prior_prob()[1])
        return self._labels[np.argmax(posterior)]






