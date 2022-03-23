"""Script to tune model hyperparameters"""
import os
from typing import List
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb  # Optional
)
from kerastuner.tuners import Hyperband
import tensorflow.keras as tfk
import tensorflow as tf
from tox21_models.utils.models_multi import (build_regression_model,  
        build_classification_model, MultiTaskModel
)

def tune_classification(datasets, save_folder: str, props: List[str], 
        dim: int = 2):
    """Tunes hyperparameters

    Args:
        props (List[str]):  
            List of properties multitask model is training on.  
            

    Returns:
        df (Dataframe):
            dataframe with results.
        best_hp_list (list):
            list of the best hyperparameters objects.
        best_dict_values (list):
            list of 5 dictionaries containing hyperparameters values.
    """
    results, best_dict_values, best_hp_list = [], [], []
    prog = 0
    for num, data in enumerate(datasets):
        PROG = 100*round(prog/5, 3)
        sys.stdout.write("\r%d%% done" % PROG)
        sys.stdout.flush()
        prog += 1
        print("Tuning dataset {}".format(num))
        directory = os.path.join(save_folder, 'hp_search', 'full', 'cv') 
        logdir = os.path.join(save_folder, 'logs', 'hparams')

        tuner = Hyperband(
            MultiTaskModel(dim, props),
            objective='val_loss',
            max_epochs=300,
            seed=10,
            directory=directory,
            project_name='fold_' + str(num),
            logger=TensorBoardLogger(
            metrics=["val_acc"], logdir=logdir
            ),
            )

        reduce_lr = tfk.callbacks.ReduceLROnPlateau(
            factor=0.6,
            monitor="val_loss",
            verbose=1,
        )
        
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                patience=30)
        
        # Create an instance of the model
        tuner.search(data['train'],
                    epochs=300,
                    validation_data=data['val'],
                    callbacks=[earlystop, reduce_lr],
                    verbose= 1
                    )

        
        best_dict_values.append(tuner.get_best_hyperparameters()[0].values) #returns dict
        best_hp_list.append(tuner.get_best_hyperparameters()[0])
        print("this is the point")
        # Post processing
        best_model = tuner.get_best_models(1)[0]
        res = np.concatenate(best_model.predict(data['val']), -1)
        
        ### This is changed##
        '''
        res = pd.DataFrame(res, columns=['pred_col0', 'pred_col1', 'pred_col2',
                'is_col0', 'is_col1', 'is_col2', 'target'
            ]
        )
        '''
#There was "target" but deleted. 
        res = pd.DataFrame(res, columns=['pred_col0', 'pred_col1',
                'is_col0', 'target'
            ]
        )
        model_folder = os.path.join(save_folder, 'models', 'full', 'cv')
        best_model.save(f'{model_folder}/{num}', include_optimizer=False)

        accurate = 0
        inaccurate = 0
        '''
        for index, row in res.iterrows():
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
        '''
        
        for index, row in res.iterrows():
            if ( row['pred_col0'] >= 0.5 and row['is_col0'] == 1):
                    accurate += 1
            elif ( row['pred_col1'] < 0.5
                and row['is_col0'] == 0):
                    accurate += 1
            else:
                inaccurate += 1
        
        results.append({'accurate': accurate, 'inaccurate': inaccurate})    
    df = pd.DataFrame(results)
    return df, best_hp_list, best_dict_values

def tune_regression(datasets, save_folder: str):
    """Tunes hyperparameters

    Args:
        regression (bool):  
            Optional. Whether a regression or classification model is being
            trained. Default is True. 
    Returns:
        dataframe with results
    """
    results, property_metric, best_values = [], [], []
    for num, data in enumerate(datasets):
        print("Tuning dataset {}".format(num))
        directory = os.path.join(save_folder, 'hp_search', 'full', 'cv') 
        logdir = os.path.join(save_folder, 'logs', 'hparams')
        tuner = Hyperband(
            build_regression_model,
            objective='val_loss',
            max_epochs=300,
            seed=20,
            directory=directory,
            project_name='fold_' + str(num),
            logger=TensorBoardLogger(
            metrics=["val_acc"], logdir=logdir
            ),
            )

        reduce_lr = tfk.callbacks.ReduceLROnPlateau(
            factor=0.8,
            monitor="val_loss",
            verbose=1,
        )
        
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                patience=30)
        
        # Create an instance of the model
        tuner.search(data['train'],
                    epochs=300,
                    validation_data=data['val'],
                    callbacks=[earlystop, reduce_lr],
                    verbose=0
                    )
        
        best_values.append(tuner.get_best_hyperparameters()[0].values)
        
        # Post processing
        best_model = tuner.get_best_models(1)[0]
        res = np.concatenate(best_model.predict(data['val']), -1)
        model_folder = os.path.join(save_folder, 'models', 'full', 'cv')
        best_model.save(f'{model_folder}/{num}', include_optimizer=False)

        _df = pd.DataFrame(res, columns=['pred', 'target', 'prop'])
        _df['prop'] = _df.prop.apply(lambda x: x.decode('utf-8'))
        props = _df.prop.unique()
        
        property_scaler = data['property_scaler']
        for prop in props:

            cond = _df[_df.prop == prop].index

            rmse_scaled = mean_squared_error(_df.loc[cond, ['target']], 
                    _df.loc[cond, ['pred']], squared=False)

            r2_scaled = r2_score(_df.loc[cond, ['target']], 
                    _df.loc[cond, ['pred']])
            
            _df.loc[cond, ['pred']] = property_scaler[prop].inverse_transform(
                    _df.loc[cond, ['pred']])

            _df.loc[cond, ['target']] = property_scaler[prop].inverse_transform(
                    _df.loc[cond, ['target']])
            
            rmse = mean_squared_error(_df.loc[cond, ['target']], _df.loc[cond,
                ['pred']], squared=False)

            r2 = r2_score(_df.loc[cond, ['target']], _df.loc[cond, ['pred']])

            property_metric.append({'prop': prop, 'rmse': rmse, 'r2':r2, 
                'fold': num, 'rmse_scaled': rmse_scaled, 
                'r2_scaled': r2_scaled})
            
        # No scaling back
        rmse = mean_squared_error(res[:,0], res[:,1], squared=False)
        r2 = r2_score(res[:,0], res[:,1])
        
        results.append({'r2': r2, 'rmse':rmse})

    df = pd.DataFrame(results)
    return df 
