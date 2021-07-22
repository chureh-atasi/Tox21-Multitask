import pytest
import numpy as np
from tox21_models.utils import data_loader
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer


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
    
    for i in range(len(df)):
        outcome = df['outcomes'].values[i]
        val = vals[i]
        expected = expected_outcome[outcome]
        assert np.array_equal(expected, val)


#def test_create_dataset():
    """Testing if create dataset returns what is expected"""
#    df = pd.DataFrame()
#    df['selector'] 
    

