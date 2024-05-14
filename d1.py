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

print("stats/measures : ", diabetes.describe())

# 0 - non-diabetic, 1 - diabetic
print(diabetes['Outcome'].value_counts()) 

#gives mean values for both these cases
print(diabetes.groupby('Outcome').mean())

#separating data and lables
X=diabetes.drop(columns='Outcome',axis=1)
Y=diabetes['Outcome']

print("dropping",X)
print("Y",Y)

#data standardization
scaler=StandardScaler()

#fitting the model
scaler.fit(X)

standardized_values=scaler.transform(X)



