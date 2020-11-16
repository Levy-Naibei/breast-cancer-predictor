""" 
    Description: This machine learning model detects breast cancer, based off of data, 
    thus, providing correct diagnosis of BC and classification of patients into malignant or benign groups
    
    Created on Sun Nov 15 23:37:40 2020 by Levy Naibei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# import dataset
df = pd.read_csv('breastcancer.csv')

# use required features
# df[['']]

""" 
    inspect dataset for empty values from each column - NaN, na, NAN
    find missing or null data points 
"""
df.isna().sum()

# remove empty values
df=df.dropna(axis=1)

# visualize B and M count
sns.countplot(df['diagnosis'], label='Count')

# define dependent and independent features
X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values

# encode catagorical data values - diagnosis
labelencoder_Y = LabelEncoder()
Y_L = labelencoder_Y.fit_transform(Y)
df["Diagnosis_encoded"]=Y_L

# split dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Using RandomForestClassification to enable class to use Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

# train the model
forest.fit(X_train, Y_train)

# accuracy of the model
Y_pred = forest.predict(X_test)
forest_score = round(accuracy_score(Y_test, Y_pred) * 100, 2)
forest_score

# save the model
pickle.dump(forest, open('bcmodel.pkl', 'wb'))