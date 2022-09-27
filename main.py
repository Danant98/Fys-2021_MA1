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

# Opening data files and transforming to numpy array
spotify_data = pd.read_csv(os.path.join("resources", "spotify_data.csv"), delimiter=",").to_numpy()
R1 = spotify_data[:, 0]
X1 = spotify_data[:, 1:]

# Estimated responses (predictions) and error (residuals) for the predictions and responses
rhat_1, error_1 = LinearRegression(X1, R1).LeaveOneOut()

# Calculating the root-mean-square error
RMSE = np.sqrt(np.sum(error_1*error_1) / len(R1))

print(RMSE)
# Array with index numbers for all songs in dataset
indexes = np.arange(0, np.shape(spotify_data)[0], 1)

# Ploting problem 1
fig, ax = plt.subplots(1, 2, tight_layout=True)
ax[0].scatter(indexes, rhat_1, label=r"$\hat{r}$")
ax[0].scatter(indexes, R1, label=r"$r$")
ax[0].vlines(indexes, rhat_1, R1, color="green")
ax[0].set_xlabel("Song by index")
ax[0].set_ylabel("Popularity of song")
ax[0].set_title("Popularity of songs by index")
ax[0].legend(loc="lower left")

ax[1].hist(error_1, color="red")
ax[1].set_title("Error/residuals")

if __name__ == '__main__':
    plt.show()
