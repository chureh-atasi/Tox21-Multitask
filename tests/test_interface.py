import pytest
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tox21_models import interface
from tox21_models.interface import (run)
from tox21_models.utils import data_loader



def test_add_fps_to_df():
    df = pd.DataFrame()
    df['SMILES'] = ["A", "C", "D"]
    fp_df = pd.DataFrame()
    fp_df['fp_1'] = [1, 2, 4]
    fp_df['fp_2'] = [3,4,8]
    fp_df['SMILES'] = ["C","D","A"]
    fp_headers = [header for header in fp_df if 'fp_' in header]
    '''for index, row in df.iterrows():
        print (fp_df.SMILES)
        print(fp_df.SMILES == row['SMILES'])
        tdf = fp_df.loc[fp_df.SMILES == row['SMILES']]
        '''
    
    print (fp_df)
    big_df = interface.add_fps_to_df (df, fp_df)
    print (" This is the new df ")
    print (big_df)
    #if (fp_df['fp_1'].all() == big_df['fp_1'].all()):
        #print ("asserting true")
        #assert True
    #else:
        #assert False

def test_run():
    np.random.seed(123)
    tf.random.set_seed(123)
    interface_folder = os.path.dirname(os.path.realpath(__file__))
    #df = pd.read_csv(os.path.join(interface_folder, 'data', 'Clean_Tox_Data.csv')) 
    df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
    df = df.iloc[0:20]
    #df = df.loc[df.prop.isin(props)
    #fp_df = pd.read_csv(os.path.join(interface_folder, 'data', 'tox21_chem_fps.csv'))
    fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
    fp_headers = [header for header in fp_df if 'fp_' in header]
    #print (fp_headers)
    print ("Debugger #1")
    df = interface.add_fps_to_df(df, fp_df, fp_headers)
    if (df.shape[0] > 5 ):
        assert True
    else:
        assert False
    #datasets_df.to_csv("CVtestdataset.csv", index=False)
    #print (datasets)

def test_avg_value():
    x = 5
    y = 6
 
test_add_fps_to_df ()
#test_run()
#test_avg_value()

