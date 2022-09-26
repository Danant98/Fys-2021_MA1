#!/usr/bin/env python
"""
Main file for assignment 1 in Fys-2021 Machine Learning. 
"""
__author__ = 'Daniel Elisabethsønn Antonsen, UiT Artic university'

# Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, LR 
import pandas as pd
from sklearn.model_selection import LeaveOneOut


# Opening data files and transforming to numpy arrays
spotify_data = pd.read_csv(os.path.join("resources", "spotify_data.csv"), delimiter=",").to_numpy()
X1 = spotify_data[:, 0]
R1 = spotify_data[:, 1:]

rhat_1 = LR.LinearRegression(X1, R1).LeaveOneOut()
print(rhat_1)


if __name__ == '__main__':
    pass
