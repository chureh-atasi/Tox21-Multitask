'''Class to laod datasets and load Forest Model '''
import pandas as pd
import sys
import os
import numpy as np
import logging
from sklearn.preprocessing import MaxAbsScaler, LabelBinarizer
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from tox21_models.forest_model import report_forest_model,Forest
from tox21_models.UMAP import UMAP_runner
from tox21_models import interface
import matplotlib.pyplot as plt

def rdkit_run():
    path = "/data/chureh/tox21_models/tox21_models/forest_model/forest_results/rdkit_results_2"
    #joblib.dump(class_LB, os.path.join(folder, "label_binarizer.pkl"), compress=3)
    df_fp = pd.read_csv("/data/chureh/tox21_models/tox21_models/data/rdkit_fp.csv", index_col = False)
    df_fp.pop("Index")
    df = pd.read_csv("/data/chureh/tox21_models/tox21_models/data/Clean_Tox_Data_3.csv")
    print('This is type of the floats' , type(df_fp.iloc[10,10]))
    print('This is type of the max floats' , type( 3.4028234e+38),  3.4028235e+38 )
    df_tot = interface.add_fps_to_df(df,df_fp)
    df_tot = df_tot.fillna(0)
    y = df_tot.pop('CHANNEL_OUTCOME')
    y_actual = df_tot.pop('CHANNEL_OUTCOME_ACTUAL')
    df_tot.pop("SMILES")
    df_tot.pop('prop')
    df_tot[df_tot==np.inf]=np.nan
    header = df_tot.columns.values.tolist()
    for x in header:
        df_tot[x].fillna(3.4028234e+38, inplace = True)
    #df_tot.fillna(df_tot.mean(), inplace=True)
    clf,y_test,y_pred,rfc_cv_score,X_test,X_train,y_train = Forest.model(df_tot,y, conf = True)
    report_forest_model.model_report(y_test,y_pred,path, label = ['active', 'inactive'])
    #report_forest_model.cv_matrix(clf,y,df,path)
    #split_active(y_pred, y_actual)

def umap_viz():
    df = pd.read_csv("/data/chureh/tox21_models/tox21_models/forest_model/final_data/df_actual_outcome_binary.csv")
    df.pop('CHANNEL_OUTCOME_ACTUAL')
    df.pop("SMILES")
    df.pop('prop')
    df = df.fillna(0)
    y = df.pop('CHANNEL_OUTCOME')
    clf,y_test,y_pred,rfc_cv_score,X_test,X_train,y_train = Forest.model(df, y)
    df_tot, class_dict = Forest.UMAP_RF_viz(X_test,X_train,y_test,y_pred,y_train)
    classes = [x for x in df_tot['umap']]
    classes = np.unique(classes)
    df_target = df_tot.pop("umap")
    UMAP_runner.umapping(df_tot,df_target,classes, class_name = class_dict)


rdkit_run()
'''
df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)
header = df.columns.values.tolist()
max_values = []
for x in header:
    values = df[x].tolist()
    max_1 = max(values)
    max_values.append(max_1)

print(max_values)
fig = plt.figure()
y_pos = np.arange(len(header))
plt.bar(y_pos, max_values, align='center', alpha=0.5)
plt.xticks(y_pos, max_values)
plt.ylabel('Max value')
plt.title('rdkit')
plt.savefig('barchart_rdkit.png', format='png')
'''
