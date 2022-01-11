from tox21_models import interface, model_reports
import pandas as pd
import numpy as np
from tox21_models.utils import data_loader
import os
import os.path

def study_txt():
    save_folder = os.getcwd()  
    directory_txt = os.path.join(save_folder, 'testcheck', "testcheck.txt")
    f= open(directory_txt,"w+")
    f.write("This is the first checkpoint")
    
def test_add_fps_to_df():
    df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
    fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
    fp_headers = [header for header in fp_df if 'fp_' in header]
    big_df = interface.add_fps_to_df (df, fp_df)
    print (" This is the new df ")
    print (big_df)
    #if (fp_df['fp_1'].all() == big_df['fp_1'].all()):
        #print ("asserting true")
        #assert True
    #else:
        #assert False

def study_report():

    
#study_txt()
test_add_fps_to_df()




