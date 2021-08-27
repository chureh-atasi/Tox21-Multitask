import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import numpy as np
import pandas as pd
from tox21_models.utils import data_loader, tuner, create_model
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import time

def add_fps_to_df(df, fp_df, cols):
    """"""
    for index, row in df.iterrows():
        new_row = {}
        for col in cols:
            new_row[col] = row[col]
       
        tdf = fp_df.loc[fp_df.SMILES == row['SMILES']]
        for col in fp_headers:
            new_row[col] = tdf[col].values[0]

        rows_with_fp.append(new_row.copy())

    print(f'Took {time.time() - st} seconds')
    del df
    df = pd.DataFrame(rows_with_fp)
    return df

def run(props: list, save_folder: str = None, regression: bool = False):
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
    df = pd.read_csv(os.path.join(interface_folder, 'data', 
            'Clean_Tox_Data.csv')) 
    df = df.loc[df.prop.isin(props)]
    # TODO REMOVE, ONLY ADDED FOR TESTING
    df = df.iloc[0:20]

    # Store fingerprints separetly to save space
    fp_df = pd.read_csv(os.path.join(interface_folder, 'data', 
            'tox21_chem_fps.csv'))

    fp_headers = [header for header in fp_df if 'fp_' in header]

    # Add fingerprints to dataframe 
    rows_with_fp = []
    cols = df.columns
    logging.info("Appending fingerprints to dataframe...")
    df = add_fps_to_df(df, fp_df, cols)
    # TODO MAKE FASTER
    st = time.time()

    # Make regression models
    if regression:
        datasets, dataset_final, test_dataset, sel_LB, class_LB = (
                data_loader.create_datasets(df, props, fp_headers)
        )

        # Check all 5 cross validations have been done, if not, run and break
        for i in range(5):
            model_path = os.path.join(cv_model_path, str(i), 'saved_model.pb')
            if not os.path.exists(model_path):
                results = tuner.tune_regression(datasets, save_folder)
                with open(os.path.join(save_folder, 'cv_results.txt'), 'w') as fin:
                    for index, row in results.iterrows():
                        fin.write(f"cv {index}: R2 = {row['r2']}, "
                                  + f"RMSE = {row['rmse']}\n")
                break

        if not os.path.exists(f'{save_folder}/chkpt'):
            model = create_model.build_regression(dataset_final, save_folder)
        else:
            model = create_model.build_regression(dataset_final, save_folder, 
                    False
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
                results = tuner.tune_classification(datasets, save_folder)
                with open(os.path.join(save_folder, 'cv_results.txt'), 'w') as fin:
                    for index, row in results.iterrows():
                        per = round(row['accurate'] 
                                    / (row['accurate'] + row['inaccurate']), 3)
                        fin.write(f"cv {index}: accurate = {row['accurate']}, "
                                  + f"Inaccruate = {row['inaccurate']}, "
                                  + f"{per*100} % successful\n")
                break

        if not os.path.exists(f'{save_folder}/chkpt'):
            model = create_model.build_classification(dataset_final, 
                    save_folder
            )
        else:
            model = create_model.build_classification(dataset_final, 
                    save_folder, False
            ) 

        val_res = np.concatenate(model.predict(dataset_final['val']), -1)
        val_res = pd.DataFrame(val_res, columns=['pred_col0', 'pred_col1', 
            'pred_col2', 'is_col0', 'is_col1', 'is_col2', 'target'
            ]
        )
        test_res = np.concatenate(model.predict(test_dataset), -1)
        test_res = pd.DataFrame(test_res, columns=['pred_col0', 'pred_col1', 
            'pred_col2', 'is_col0', 'is_col1', 'is_col2', 'target'
            ]
        )
        accurate = 0
        inaccurate = 0
        for index, row in val_res.iterrows():
            if (row['pred_col0'] > row['pred_col2']    
                and row['pred_col0'] > row['pred_col1']
                and row['is_col0'] == 1):
                    accurate += 1
            elif (row['pred_col2'] > row['pred_col0']    
                and row['pred_col2'] > row['pred_col1']
                and row['is_col2'] == 1):
                    accurate += 1
            elif (row['pred_col1'] > row['pred_col0']    
                and row['pred_col2'] < row['pred_col1']
                and row['is_col1'] == 1):
                    accurate += 1
            else:
                inaccurate += 1


        with open(os.path.join(save_folder, 'val_results.txt'), 'w') as fin:
            fin.write(f"Accurate: {accurate}\nInaccurate: {inaccurate}\n")
            per = round(accurate/(accurate + inaccurate), 3)*100
            fin.write(f"{per} % successful\n")
        
        accurate = 0
        inaccurate = 0
        for index, row in test_res.iterrows():
            if (row['pred_col0'] > row['pred_col2']    
                and row['pred_col0'] > row['pred_col1']
                and row['is_col0'] == 1):
                    accurate += 1
            elif (row['pred_col2'] > row['pred_col0']    
                and row['pred_col2'] > row['pred_col1']
                and row['is_col2'] == 1):
                    accurate += 1
            elif (row['pred_col1'] > row['pred_col0']    
                and row['pred_col2'] < row['pred_col1']
                and row['is_col1'] == 1):
                    accurate += 1
            else:
                inaccurate += 1


        with open(os.path.join(save_folder, 'test_results.txt'), 'w') as fin:
            fin.write(f"Accurate: {accurate}\nInaccurate: {inaccurate}\n")
            per = round(accurate/(accurate + inaccurate), 3)*100
            fin.write(f"{per} % successful\n")
        
