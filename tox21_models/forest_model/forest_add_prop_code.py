import os 
import sys
from tox21_models.forest_model import Forest
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer
import logging
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("df_tot.csv")
print(df.shape[1])
st = time.time()
prog = 0
for index, row in df.iterrows():
    PROG = 100*round(prog/len(df), 3)
    sys.stdout.write("\r%d%% done" % PROG)
    sys.stdout.flush()
    prog += 1
    if (row['prop'] == 'Liver'):
        df['is_liver']  = 1
        df['is_kidney']  = 0
        df['is_chicken']  = 0
        df['is_Hamster']  = 0
        df['is_Apoptosis']  = 0
    elif (row['prop'] == 'Kidney'):
        df['is_liver']  = 0
        df['is_kidney']  = 1
        df['is_chicken']  = 0
        df['is_Hamster']  = 0
        df['is_Apoptosis']  = 0
    elif (row['prop'] == 'Chicken'):
        df['is_liver']  = 0
        df['is_kidney']  = 0
        df['is_chicken']  = 1
        df['is_Hamster']  = 0
        df['is_Apoptosis']  = 0
    elif (row['prop'] == 'Hamster'):
        df['is_liver']  = 0
        df['is_kidney']  = 0
        df['is_chicken']  = 0
        df['is_Hamster']  = 1
        df['is_Apoptosis']  = 0
    elif (row['prop'] == 'Apoptosis'):
        df['is_liver']  = 0
        df['is_kidney']  = 0
        df['is_chicken']  = 0
        df['is_Hamster']  = 0
        df['is_Apoptosis']  = 1

print (df.shape[1])
Forest.model(df)
