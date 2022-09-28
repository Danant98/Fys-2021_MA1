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

# Removing popularity values less than 80
R1_2 = R1[R1 > 80]
X1_2 = X1[R1 > 80]

# Estimated responses (predictions) and error (residuals) for the predictions and responses
rhat_1, error_1 = LinearRegression(X1, R1).LeaveOneOut()
rhat_12, error_12 = LinearRegression(X1_2, R1_2).LeaveOneOut()

# Calculating the root-mean-square error
RMSE = np.sqrt(np.sum(error_1*error_1) / len(R1))
# Calculating R^2 
SSRES = np.sum(np.square(error_1))
SSRTOT = np.sum(np.square(R1 - np.mean(R1)))
Rsqared = 1 - (SSRES / SSRTOT)

# Calculate the root-mean-square error for the predictions 
RMSE2 = np.sqrt(np.sum(error_12*error_12) / len(R1_2))
# Calculating R^2 
SSRES2 = np.sum(error_12*error_12)
SSRTOT2 = np.sum((R1_2 - np.mean(R1_2))**2)
Rsqared2 = 1 - (SSRES2 / SSRTOT2)


# Printing the value for RMSE and R^2
#print("Root-mean-squere error is given as {0:.6f} \nR^2 is given as {1:.6f}".format(RMSE, Rsqared))

# Array with index numbers for all songs in dataset
indexes = np.arange(0, np.shape(R1)[0], 1)
indexes_1 = np.arange(0, np.shape(R1_2)[0], 1)

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
ax[1].set_xlabel("Error or residuals")
ax[1].set_title("Error/residuals for songs")

# Ploting problem 1 with all values below 80 cut out
fig1, ax1 = plt.subplots(1, 2, tight_layout=True)
ax1[0].scatter(indexes_1, rhat_12, label=r"$\hat{r}$")
ax1[0].scatter(indexes_1, R1_2, label=r"$r$")
ax1[0].vlines(indexes_1, rhat_12, R1_2, color="green")
ax1[0].set_xlabel("Song by index")
ax1[0].set_ylabel("Popularity of song")
ax1[0].set_title("Popularity of songs by index for songs with popularity above 80")
ax1[0].legend(loc="lower left")

ax1[1].hist(error_12, color="red")
ax1[1].set_xlabel("Error or residuals")
ax1[1].set_title("Error/residuals for songs with popularity above 80")

#print("Root-mean-squere error without values below 80 is given as {0:.6f} \nR^2 without values below 80 is given as {1:.6f}".format(RMSE2, Rsqared2))



if __name__ == '__main__':
    plt.show()
