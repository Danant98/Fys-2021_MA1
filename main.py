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

# Residuals or errors
error_1 = R1 - rhat_1

# Array with index numbers for all songs in dataset
indexes = np.arange(0, np.shape(spotify_data)[0], 1)

# Ploting problem 1
fig, ax = plt.subplots(1, 2)
ax[0].scatter(indexes, rhat_1, label=r"$\hat{r}$")
ax[0].scatter(indexes, R1, label=r"$r$")
ax[0].set_xlabel("Song by index")
ax[0].set_ylabel("Popularity of song")
ax[0].legend(loc="lower left")

ax[1].hist(error_1, indexes)


if __name__ == '__main__':
    plt.show()
