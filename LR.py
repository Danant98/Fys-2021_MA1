"""
File containg the multivaritate linear regression class
"""
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

    def __bestmodel(self):
        """
        Method for calculating the best fit model (g(x|theta))
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(self._X.T, self._X)), self._X.T), self._r)
    
    def linear_regression(self):
        """
        Method for calculating the linear regression using best model
        """
        # Using the best fit model to calculate the linear regression 
        try:
            BP = self.__bestmodel()
            return np.dot(self._X, BP)
        # Caching errors if method does not work
        except Exception as e:
            print("OPS! Something went wrong. Check {}\n for details.".format(str(e)))


