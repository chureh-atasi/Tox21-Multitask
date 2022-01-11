import pytest
import numpy as np
from tox21_models.utils import data_loader
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from tox21_models import interface


def test_assertTrue():
    assert True
    
def test_label_binarizer():
    df = pd.DataFrame()
    df['outcomes'] = ['inactivate', 'agonist', 'inactivate', 'antagonist', 
            'inactivate', 'agonist', 'antagonist']
    class_LB = LabelBinarizer()
    vals = class_LB.fit_transform(df['outcomes'].values)
    expected_outcome = {'inactivate': [0, 0, 1], 'agonist': [1, 0, 0],
            'antagonist': [0, 1, 0]}
    print(vals)
    for i in range(len(df)):
        outcome = df['outcomes'].values[i]
        print (outcome)
        val = vals[i]
        expected = expected_outcome[outcome]
        assert np.array_equal(expected, val)


def test_create_datasets():
    """Testing if create dataset returns what is expected"""
    #df = pd.DataFrame()
    #df['selector'] 
    skf = StratifiedKFold(n_splits=5)
    props = ['Liver']
    df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
    df = df.iloc[0:20]
    fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
    fp_headers = [header for header in fp_df if 'fp_' in header]
    df = interface.add_fps_to_df(df, fp_df, fp_headers)
    training_df, test_df = train_test_split(df, test_size=0.1, 
            stratify=df.prop, random_state=123)
    #training_df, validation_df = train_test_split(training_df, test_size=0.2, 
            #stratify=training_df.prop, random_state=123)
    #training_df, validation_df = training_df.copy(), validation_df.copy()
    print (" Elements in the training set ", training_df.shape[0])
    print (training_df.prop)
    for train_index, val_index in skf.split(training_df, training_df.prop):
        print ("train index  val_index " , train_index , val_index)

    #datasets, dataset_final, test_dataset, sel_LB, class_LB = (
    #           data_loader.create_datasets(df, props, fp_headers, False))

    #print (len(datasets))
    #assert (len(datasets) > 20
    

def test_create_dataset_dict():
    "Tests dictionoary method"
    


test_create_datasets()
#test_label_binarizer()
    

