#!/usr/bin/env python
"""
Main file for assignment 1 in Fys-2021 Machine Learning. 
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from LR import LinearRegression


# Opening data files and transforming to numpy arrays
spotify_data = pd.read_csv(os.path.join("resources", "spotify_data.csv"), delimiter=",").to_numpy()
R1 = spotify_data[:, 0]
X1 = spotify_data[:, 1:]

# Estimated responses (predictions)
rhat_1 = LinearRegression(X1, R1).LeaveOneOut()

indexes = np.arange(0, np.shape(spotify_data)[0], 1)

plt.plot(indexes, rhat_1)
plt.plot(indexes, R1)
plt.show()

if __name__ == '__main__':
    pass
