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
from Bayes import BayesClassifier

"""
### Problem 1
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

sns.histplot(error_1, ax=ax[1], color="red")
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

sns.histplot(error_12, ax=ax1[1], color="red")
ax1[1].set_xlabel("Error or residuals")
ax1[1].set_title("Error/residuals for songs with popularity above 80")

#print("Root-mean-squere error without values below 80 is given as {0:.6f} \nR^2 without values below 80 is given as {1:.6f}".format(RMSE2, Rsqared2))
"""
### Problem 2

# Opening data for problem 2
datatrain = np.genfromtxt(os.path.join("resources", "optdigits-1d-train.csv"))

def split_data(data:np.ndarray):
    """
    Function to split a single dataset into 2 classes

    Input: 
        data:np.ndarray, original dataset
        
    Output: 
        X1: np.ndarray, class 1 
        X2: np.ndarray, class 2    
    """
    X = data[:, 1]
    labels = data[:, 0]
    X1 = X[labels < 1]
    X2 = X[labels > 0]

    return X1, X2, labels

# Splitting dataset into 2 
X1, X2, labels  = split_data(datatrain)

# Run BayesClassifier class
bayes = BayesClassifier(X1, X2)

# Calculating beta, mean and variance
beta = bayes.beta_hat()
mu = bayes.mean()
sigma_2 = bayes.variance()

# Calculating the prior probabilities for C0 and C1
PC0 = len(X1) / len(labels)
PC1 = len(X2) / len(labels)
#print("P(C0) = " + str(PC0))
#print("P(C1) = " + str(PC1))

# Defining 
t1 = np.linspace(0, 1, len(X1))
t2 = np.linspace(0, 1, len(X2))

# Calling normal and gamma distribution functions
g = bayes.Gamma(t1) 
n = bayes.Normal(t2)

# Ploting the data from
#fig2, ax2 = plt.subplots(2, 1, tight_layout=True)
sns.histplot(datatrain[:, 1], bins=50, stat="probability")
plt.scatter(t1, g/100, color="green", label="Gamma distribution")
plt.scatter(t2, n/100, color="red", label="Normal distribution")
plt.xlabel("x")
plt.title("Plot of probability distributions")
plt.legend()




if __name__ == '__main__':
    plt.show()
