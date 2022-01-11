import seaborn as sns
import pandas as pd
import numpy as np
import os
from tox21_models.utils import data_loader
from tox21_models import interface

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#We delete the first 20 rows of the example just to get a dataset that works better for this example
#df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
#fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
#X = interface.add_fps_to_df(df, fp_df)
#X.to_csv("Full dataset.csv")

X = pd.read_csv("Full dataset.csv")
y = X.pop('CHANNEL_OUTCOME')
z = X.pop("SMILES")
k = X.pop("prop")

#We split nonrandomly
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, shuffle=False)


#We annotate the training data with the 'train' label in a new column
X_train = X_train.assign(dataset='train')

#...and add back the target variable. It's natural to use it in the random forest classifier
X_train = pd.concat([X_train, y_train], axis = 1)

#We do the same with the test data
X_test = X_test.assign(dataset='test')
X_test = pd.concat([X_test, y_test], axis = 1)

#We add everything together
X_rf = pd.concat([X_train, X_test])

#We encode the old target variable for the classifier
X_rf = pd.get_dummies(X_rf, columns=['CHANNEL_OUTCOME'])

#The dataset column - that is, whether the datapoint belongs to the training or the test dataset - is exactly what we are trying to predict
y_rf = X_rf.pop('dataset')
print (X_rf)

#The following is just a good and old Random Forest classifier
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
     X_rf, y_rf, test_size=0.33, random_state=42)

print (type(X_rf_train.values))

clf = RandomForestClassifier(max_depth=2, random_state=42)
clf.fit(X_rf_train.values, y_rf_train.values)

print("Score of the classifier with nonrandom train/test split: ",
        clf.score(X_rf_test.values, y_rf_test.values))

###########################################################################################3

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.45, random_state = 42)

#From here the code is exactly the same as before

X_train = X_train.assign(dataset='train')
X_train = pd.concat([X_train, y_train], axis = 1)

X_test = X_test.assign(dataset='test')
X_test = pd.concat([X_test, y_test], axis = 1)


X_rf = pd.concat([X_train, X_test])
X_rf = pd.get_dummies(X_rf, columns=['CHANNEL_OUTCOME'])

y_rf = X_rf.pop('dataset')

#The stratify parameter will keep the distribution of y_rf as in y
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
     X_rf, y_rf, test_size=0.33, random_state=42, stratify = y_rf)


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_rf_train, y_rf_train)

print("Score of the classifier with random train/test split:    ", 
        clf.score(X_rf_test.values, y_rf_test.values))



