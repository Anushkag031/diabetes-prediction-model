import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection and analysis
#loading the datsets

diabetes = pd.read_csv('/Users/anushkag031/Documents/machinelearning/diab-pred/diabetes.csv')

#printing the rows
print(diabetes.head())
#entries in the model
print(diabetes.shape)
