"""
Main file for assignment 1 in Fys-2021 Machine Learning. 

@author: Daniel Elisabethsønn Antonsen, UiT Arctic university.
"""
# Import libraries and modules
import numpy as np
import seaborn as sns
import os, LR 

# Opening data files and transforming to numpy arrays
spotify_data = np.genfromtxt(os.path.join("resources", "spotify_data.csv"), delimiter=",")
# Trening dataset for spotify dataset, using the N-1 first values 
rtrening_1 = spotify_data[:np.shape(spotify_data)[0] - 1, 0]
xtrening_1 = spotify_data[:np.shape(spotify_data)[0] - 1, 1:]
# Test dataset using cross-validation, dataset contains the N element from original dataset 
rtest_1 = spotify_data[-1, 0]
xtest_1 = spotify_data[-1, 1:]
# Running linear regression for spotify dataset
rhat_1 = LR.LinearRegression(xtrening_1, rtrening_1)



if __name__ == '__main__':
    pass
