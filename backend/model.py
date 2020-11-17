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

#use required features
cdf = df[['diagnosis', 'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

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
X = cdf.iloc[:, 1:11].values
Y = cdf.iloc[:, 0:1].values

# encode catagorical data values - diagnosis
le = LabelEncoder()
Y_L = le.fit_transform(Y.ravel())

cdf["diagnosis_encoded"]=Y_L

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