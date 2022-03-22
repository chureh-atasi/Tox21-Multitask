import os
import logging
import copy
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import pandas as pd
from tox21_models.utils import data_loader, tuner_multi, create_model_multi
from tox21_models import model_reports
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from kerastuner import HyperParameters
import time
import sys
import json
import joblib

def add_fps_to_df(df, fp_df):
    """Add the dataset fingerprint to the chemical dataframe

    Args:  
        df (dataframe):  
            the dataframe that contains all the chemicals and their toxicity values.  
        fp_df (dataframe):  
            The dataframe that contains the chemicals fingerprint to the dataframe (df).      

    Return a dataframe that contains the chemicals along with their fingerprints.  
    """
    fp_df = fp_df.set_index(keys='SMILES')
    fp_dict = fp_df.to_dict(orient='index') 
    
    st = time.time()
    cols = df.columns
    fp_cols = fp_df.columns
    prog = 0
    #df_tot = pd.DataFrame()
    tot_dat = []
    for index, row in df.iterrows():
        PROG = 100*round(prog/len(df), 3)
        sys.stdout.write("\r%d%% done" % PROG)
        sys.stdout.flush()
        prog += 1
        new_row = {}
        for col in cols:
            new_row[col] = row[col]
       
        #new_row = pd.DataFrame(new_row, index = ['a'])
        #tdf = fp_df.loc[fp_df.SMILES == row['SMILES']]
        fp = fp_dict[row['SMILES']]
        #new_row = pd.merge(new_row, tdf, on="SMILES")
        new_row.update(fp)
        tot_dat.append(copy.deepcopy(new_row))
        #df_tot = pd.concat([df_tot, new_row] , axis = 0)
    df_tot = pd.DataFrame(tot_dat)
    print(f'Took {time.time() - st} seconds')
    return df_tot

def run(props: list, save_folder: str = None, regression: bool = False, dim: int = 2):
    """Run meta model cross validation and training
    
    Args:  
        props (list):  
            List of properties (props) to create model for  
        save_folder (str):  
            Optional. Location for save folder (else, saved in current working
            directory.  
        regression (bool):  
            Optional. Whether a regression or classification model is being
            trained. Default is True. 
    """
    if save_folder is None:
        save_folder = os.getcwd()
    cv_model_path = os.path.join(save_folder, 'models', 'full', 'cv')

    np.random.seed(123)
    tf.random.set_seed(123)
    interface_folder = os.path.dirname(os.path.realpath(__file__))

    #df = pd.read_csv("/data/chureh/tox21_models/data_clean/deeptox_fp.csv")
    #df = pd.read_csv("/data/chureh/tox21_models/tox21_models/forest_model/final_data/df_actual_outcome_binary.csv")


    df = pd.read_csv(os.path.join(interface_folder, 'data', 
            'Clean_Tox_Data.csv')) 
    df = df.loc[df.prop.isin(props)]
    #df = df.iloc[0:100]
    

    # Store fingerprints separetly to save space
    fp_df = pd.read_csv(os.path.join(interface_folder, 'data', 
            'tox21_chem_fps.csv'))

    
    # Add fingerprints to dataframe 
    logging.info("Appending fingerprints to dataframe...")
    df = add_fps_to_df(df, fp_df)
    df.to_csv(os.path.join(interface_folder, 'data', 'temp.csv'))
    quit()
    #df = pd.read_csv(os.path.join(interface_folder, 'data', 'temp.csv'))
    # TODO MAKE FASTER
    # Done
    
    fp_headers = [header for header in df if 'fp_' in header]
    print ("This is length " , len(fp_headers))
    '''
    fp_headers = [header for header in df]
    identifiers = ["SMILES","CHANNEL_OUTCOME","prop","CHANNEL_OUTCOME_ACTUAL","sample_ID"]
    for element in identifiers:
        if element in fp_headers:
            fp_headers.remove(element)
    '''
    
    #df = pd.read_csv("Full dataset.csv")
    #df = pd.read_csv("/data/chureh/tox21_models/tox21_models/forest_model/final_data/df_actual_outcome_binary.csv")
    #df = pd.read_csv("/data/chureh/tox21_models/data_clean/df_var_reduced_feature.csv")
    print("dataset is loaded")
    df.pop('CHANNEL_OUTCOME_ACTUAL')

    # Make regression models
    if regression:
        datasets, dataset_final, test_dataset, sel_LB, class_LB = (
                data_loader.create_datasets(df, props, fp_headers)
        )

        # Check all 5 cross validations have been done, if not, run and break
        for i in range(5):
            model_path = os.path.join(cv_model_path, str(i), 'saved_model.pb')
            if not os.path.exists(model_path):
                results, best_hp = tuner_multi.tune_regression(datasets, save_folder)
                with open(os.path.join(save_folder, 'cv_results.txt'), 'w') as fin:
                    for index, row in results.iterrows():
                        fin.write(f"cv {index}: R2 = {row['r2']}, "
                                  + f"RMSE = {row['rmse']}\n")
                break

        if not os.path.exists(f'{save_folder}/chkpt'):
            model = create_model_multi.build_regression(dataset_final, save_folder)
        else:
            model = create_model_multi.build_regression(dataset_final, save_folder, 
                    False, hp = best_hp
            ) 

        val_res = np.concatenate(model.predict(dataset_final['val']), -1)
        val_res = pd.DataFrame(val_res, columns=['pred', 'target', 'prop'])
        test_res = np.concatenate(model.predict(test_dataset), -1)
        test_res = pd.DataFrame(test_res, columns=['pred', 'target', 'prop'])


        plt.scatter(val_res['pred'], val_res['target'])
        img = os.path.join(save_folder, 'val_result.png')
        plt.savefig(img)
        plt.scatter(test_res['pred'], test_res['target'])
        img = os.path.join(save_folder, 'test_result.png')
        plt.savefig(img)
    
    # Make classification models
    else:
        datasets, dataset_final, test_dataset, sel_LB, class_LB = (
                data_loader.create_datasets(df, props, fp_headers, regression)
        )
        
        # Check all 5 cross validations have been done, if not, run and break
        for i in range(5):
            model_path = os.path.join(cv_model_path, str(i), 'saved_model.pb')
            if not os.path.exists(model_path):
                results, best_hp_list, best_hp_values = tuner_multi.tune_classification(datasets, save_folder)
                with open(os.path.join(save_folder, 'cv_results.txt'), 'w') as fin:
                    for index, row in results.iterrows():
                        per = round(row['accurate'] 
                                    / (row['accurate'] + row['inaccurate']), 3)
                        fin.write(f"cv {index}: accurate = {row['accurate']}, "
                                  + f"Inaccruate = {row['inaccurate']}, "
                                  + f"{per*100} % successful\n")
                break
        results, best_hp_list, best_hp_values = tuner_multi.tune_classification(datasets, save_folder)
        with open(os.path.join(save_folder, 'cv_results.txt'), 'w') as fin:
            for index, row in results.iterrows():
                per = round(row['accurate'] 
                    / (row['accurate'] + row['inaccurate']), 3)
                fin.write(f"cv {index}: accurate = {row['accurate']}, "
                            + f"Inaccruate = {row['inaccurate']}, "
                            + f"{per*100} % successful\n")
        path = os.path.join(save_folder, 'testcheck')
        directory_txt = os.path.join(path, "testcheck_1.txt")
        f= open(directory_txt,"w+")
        #:wqf.write("This is the first checkpoint in interface right after it's done hyperparamater bla bla ")
        with open(directory_txt,'w+') as f_1:
            for element in best_hp_values:
                f_1.write(json.dumps(element))
        f.close()
        
        
        avg_hp_dict = {}
        for key in best_hp_values[0]:
            if key != ("tuner/trial_id"):
                tot = best_hp_values[0][key] +  best_hp_values[1][key] +  best_hp_values[2][key] + best_hp_values[3][key] +  best_hp_values[4][key]
                if (isinstance(tot, int)):
                    best_avg = round(tot/5)
                else:
                    best_avg = tot/5 
                avg_hp_dict[key] = best_avg
        
    
        
        if not os.path.exists(f'{save_folder}/chkpt'):
            model = create_model_multi.build_classification(dataset_final, 
                    save_folder, best_hp_list[1]
            )
        else:
            model = create_model_multi.build_classification(dataset_final, 
                    save_folder, best_hp_list[1], False
            ) 

        val_res = np.concatenate(model.predict(dataset_final['val']), -1)
        val_res = pd.DataFrame(val_res, columns=['pred_col0', 'pred_col1', 
         'is_col0', 'target'
            ]
        )
        test_pred = model.predict(test_dataset)
        test_res_1 = np.concatenate(model.predict(test_dataset), -1)
        print ("###This is test res##### \n \n ")
        print(('\n') , ([[item for item in row] for row in test_pred]))
        print(' \n \n ###This is the end of test res###')
        test_res = pd.DataFrame(test_res_1, columns=['pred_col0', 'pred_col1', 
             'is_col0', 'target'
            ]
        )
        path = os.path.join(save_folder, 'testcheck')
        directory_txt = os.path.join(path, "testcheck_4.txt")
        f= open(directory_txt,"w+")
        f.write("This is the checkpoint in interface right after it adds the columns ")
        with open(directory_txt,'w+') as f_1:
            for element in best_hp_values:
                f_1.write(json.dumps(element))
        f.close()
        accurate = 0
        inaccurate = 0
        for index, row in val_res.iterrows():
            if ( row['pred_col0'] >= 0.5 and row['is_col0'] == 1):
                    accurate += 1
            elif ( row['pred_col1'] < 0.5
                and row['is_col0'] == 0):
                    accurate += 1
            else:
                inaccurate += 1


        with open(os.path.join(save_folder, 'val_results_1.txt'), 'w') as fin:
            fin.write(f"Accurate: {accurate}\nInaccurate: {inaccurate}\n")
            per = round(accurate/(accurate + inaccurate), 3)*100
            fin.write(f"{per} % successful\n")
        
        accurate = 0
        inaccurate = 0
        for index, row in test_res.iterrows():
            if ( row['pred_col0'] >= 0.5 and row['is_col0'] == 1):
                    accurate += 1
            elif ( row['pred_col1'] < 0.5
                and row['is_col0'] == 0):
                    accurate += 1
            else:
                inaccurate += 1
        
        with open(os.path.join(save_folder, 'test_results.txt'), 'w') as fin:
            fin.write(f"Accurate: {accurate}\nInaccurate: {inaccurate}\n")
            per = round(accurate/(accurate + inaccurate), 3)*100
            fin.write(f"{per} % successful\n")
        
        
        # 0 = agonist
        # 1 = antagonist
        # 2 = inactive
        #with open(os.path.join(os.path.dirname(interface_folder),
           # 'label_binarizer.pkl'), 'rb') as f:
          #  l_b = joblib.load(f)
        labels = [0,1]
        #self.Y_class = self.l_b.inverse_transform(np.stack(self.y_test))
        #self.pred = self.l_b.inverse_transform(np.stack(pred_y)
        path = os.path.join(save_folder, 'testcheck')
        directory_txt = os.path.join(path, "testcheck_5.txt")
        f= open(directory_txt,"w+")
        f.write("This is the checkpoint in interface right after it adds the columns ")
        val_matrix = model_reports.build_matrix(val_res, labels) 
        test_matrix = model_reports.build_matrix(test_res, labels) 
        path = os.path.join(save_folder, 'testcheck')
        directory_txt = os.path.join(path, "testcheck_6s.txt")
        f= open(directory_txt,"w+")
        f.write("This is the checkpoint in interface right after it adds the columns ")
        val_matrix_path = os.path.join(save_folder, 'validation_matrix_with_lb.csv')
        test_matrix_path = os.path.join(save_folder, 'test_matrix_with_lb.csv')
        #print (test_res.shape[0])

        val_matrix.to_csv(val_matrix_path, index=False)
        test_matrix.to_csv(test_matrix_path, index = False)
