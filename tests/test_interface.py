import pytest
import pandas as pd
from tox21_models import interface

def test_add_fps_to_df():
    df = pd.DataFrame()
    df['SMILES'] = ["C", "D", "A"]
    fp_df = pd.DataFrame()
    fp_df['fp_1'] = [1, 2, 4]
    fp_df['SMILES'] = ["C","D","A"]
    fp_headers = [header for header in fp_df if 'fp_' in header]
    #for index, row in df.iterrows():
        #print (fp_df.SMILES)
        #print(fp_df.SMILES == row['SMILES'])
        #tdf = fp_df.loc[fp_df.SMILES == row['SMILES']]
    
    #print (tdf)
    big_df = interface.add_fps_to_df (df, fp_df, fp_headers)
    print (big_df)
    if (fp_df['fp_1'].all() == big_df['fp_1'].all()):
        print ("asserting true")
        assert True
    else:
        assert False

 
test_add_fps_to_df ()


