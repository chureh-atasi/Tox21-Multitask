from tox21_models.utils import data_loader
from tox21_models import interface
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer

def study_data():
    props = ['Liver']

    df = pd.DataFrame()
    df['SMILES'] = ["C", "D", "A"]
    fp_df = pd.DataFrame()
    fp_df['fp_1'] = [1, 2, 4]
    fp_df['SMILES'] = ["C","D","A"]
    df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
    df = df.iloc[0:20]
    fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
    fp_headers = [header for header in fp_df if 'fp_' in header]
    df = interface.add_fps_to_df(df, fp_df)
    datasets, dataset_final, test_dataset, sel_LB, class_LB = (
        data_loader.create_datasets(df, props, fp_headers, False))
    i = 0
    for num, data in enumerate(dataset_final):
        #print(i)
        #i = i + 1
        #print ("this is num")
        #print (num)
        print (data)
    #print (dataset_final["train"])

def study_createdataset():
    df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data.csv")
    df = df.iloc[0:20]
    fp_df = pd.read_csv(r"/data/chureh/tox21_models/tox21_models/data/tox21_chem_fps.csv")
    fp_headers = [header for header in fp_df if 'fp_' in header]
    df = interface.add_fps_to_df(df, fp_df)
    sel_LB = LabelBinarizer()
    selectors = sel_LB.fit_transform(df['prop'].values)
    df['selector'] = selectors
    class_LB = LabelBinarizer()
    vals = class_LB.fit_transform(df['CHANNEL_OUTCOME'].values)
    df['class_vals'] = [val for val in vals]
    labels_0 = []
    for i in range (len(vals[0])):
        labels_0.append(i)
    print(labels_0)
    print("This is type" , vals[0], type(class_LB.inverse_transform(vals)))
    #print ("this is class LB" , class_LB.inverse_transform(vals[0,1]))
    dataset = data_loader.create_dataset(df,False,fp_headers)
    #print (dataset)

#study_data()
study_createdataset()



