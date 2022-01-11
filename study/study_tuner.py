from tox21_models.utils import tuner, data_loader
import pandas as pd
import os
from tox21_models import interface

def study_tune_class():
    props = ['Liver']
    save_folder = os.getcwd()
    df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
    df = df.iloc[0:20]
    fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
    fp_headers = [header for header in fp_df if 'fp_' in header]
    df = interface.add_fps_to_df(df, fp_df, fp_headers)
    datasets, dataset_final, test_dataset, sel_LB, class_LB = (
            data_loader.create_datasets(df, props, fp_headers, False))
    results, best_hp_list, best_hp_dict = tuner.tune_classification(datasets,save_folder )
    save_folder = os.getcwd()  
    directory_txt = os.path.join(save_folder, 'testcheck', "testcheck.txt")
    f= open(directory_txt,"w+")
    f.write(" this is to show that it's done ")
    #for element in best_hp_dict:
    #    f.write(element)
    #    f.write('\n')
    f.close()
    print (" the list is  " , best_hp_dict)
    print (" this is the dictionary  ", best_hp_dict[0])

study_tune_class()

