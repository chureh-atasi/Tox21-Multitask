from typing import List

import os
from kerastuner import HyperParameters

import tensorflow.keras as tfk
import tensorflow as tf
import json
from tox21_models.utils.models_multi import (build_regression_model,  
        build_classification_model, MultiTaskModel
)
import tensorflow_recommenders as tfrs

def build_classification(dataset, save_folder, hp_dict, props: List[str],
        build_new: bool = True, dim: int = 2):
    """Builds a model and saves it

    Args:
        dataset (dict):
            {'train': tf.data.Dataset, 'val': tf.data.Dataset,
             'property_scaler': sklearn.preprocessing.MinMaxScaler}
             Where train is the training dataset, 
             and val is the validation dataset
        build_new (bool):
            If true, build new model, else, load checkpoint
        hp_dict (dict):
            a dictionary containing the averaged hyperparamater values.
        props (List[str]):  
            List of properties multitask model is training on.  
    
    Returns:
        Build model
    """

    #print (" This is in the create_model file")
    path = os.path.join(save_folder, 'testcheck')
    directory_txt = os.path.join(path, "testcheck_2.txt")
    f= open(directory_txt,"w+")
    #:wqf.write("This is the first checkpoint in create model before it goes through if")
    #with open(directory_txt,'w') as f_1:
        #f_1.write(json.dumps(hp_dict))

    f.close()

    if hp_dict == 0:
        hp = HyperParameters()
        for prop in ['input'] + props:
            hp.Fixed(f'{prop}_units_0', value=100)
            hp.Fixed(f'{prop}_units_1', value=100)
            hp.Fixed(f'{prop}_units_2', value=100)
            hp.Fixed(f'{prop}_num_layers', value=3)

            hp.Fixed(f'{prop}_dropout_0', value=0)
            hp.Fixed(f'{prop}_dropout_1', value=0)
            hp.Fixed(f'{prop}_dropout_2', value=0)
    
    else:
        hp = hp_dict
    '''
    else:
        hp = HyperParameters()
        hp.Fixed('learning_rate', value = hp_dict['learning_rate']) #add the hp_dict here
        hp.Fixed('concat_at', value = hp_dict['concat_at'])
        hp.Fixed('units_0', value= hp_dict['units_0'])
        hp.Fixed('units_1', value= hp_dict['units_1'])
        hp.Fixed('units_2', value= hp_dict['units_2'])

        hp.Fixed('dropout_0', value= hp_dict['dropout_0'])
        hp.Fixed('dropout_1', value= hp_dict['dropout_1'])
        hp.Fixed('dropout_2', value= hp_dict['dropout_2'])
        hp.Fixed('num_layers', value =hp_dict['num_layers'] )
        hp.Fixed('tuner/epochs', value =hp_dict['tuner/epochs'] )
        hp.Fixed('tuner/initial_epoch', value =hp_dict['tuner/initial_epoch'] )
        hp.Fixed('tuner/bracket', value =hp_dict['tuner/bracket'] )
        hp.Fixed('tuner/round', value =hp_dict['tuner/round'] )
        '''

    directory_txt = os.path.join(path, "testcheck_3.txt")
    f= open(directory_txt,"w+")
    f.write("This is the first checkpoint in create model after it goes through if")
    f.close()
    # Create an instance of the model
    hypermodel = MultiTaskModel(dim, props)
    model = hypermodel.build(hp=hp) 
    directory = os.path.join(save_folder, 'chkpt')
    if build_new:
        checkpoint = tfk.callbacks.ModelCheckpoint(
                f'{directory}/model_test', monitor='val_loss', verbose=1, 
                save_best_only=True, save_weights_only=True, mode='auto'
        )

        reduce_lr = tfk.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,
                    cooldown=0,
                    verbose=1,
                    )
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                patience=30)

        hist = model.fit(dataset['train'], epochs=100, 
                validation_data=dataset['val'], callbacks=[earlystop, 
                checkpoint, reduce_lr])
    else:
        model.fit(dataset['train'], epochs=1,validation_data=dataset['val'])
        model.load_weights(tf.train.latest_checkpoint(directory))
    return model

def build_regression(dataset, save_folder, build_new: bool = True):
    """Builds a model and saves it

    Args:
        dataset (dict):
            {'train': tf.data.Dataset, 'val': tf.data.Dataset,
             'property_scaler': sklearn.preprocessing.MinMaxScaler}
             Where train is the training dataset, 
             and val is the validation dataset
        build_new (bool):
            If true, build new model, else, load checkpoint
    
    Returns:
        Build model
    """
    hp = HyperParameters()
    hp.Fixed('learning_rate', value=1e-3)
    hp.Fixed('concat_at', value=1)
    hp.Fixed('units_0', value=400)
    hp.Fixed('units_1', value=400)
    hp.Fixed('units_2', value=400)

    hp.Fixed('dropout_0', value=0.1)
    hp.Fixed('dropout_1', value=0.1)
    hp.Fixed('dropout_2', value=0.1)

    # Create an instance of the model
    model = build_regression_model(hp)
    directory = os.path.join(save_folder, 'chkpt')
    if build_new:
        checkpoint = tfk.callbacks.ModelCheckpoint(
                f'{directory}/model_test', monitor='val_loss', verbose=1, 
                save_best_only=True, save_weights_only=True, mode='auto'
        )

        reduce_lr = tfk.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.95,
                    cooldown=0,
                    verbose=1,
                    )
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", 
                patience=30)

        hist = model.fit(dataset['train'], epochs=20, 
                validation_data=dataset['val'], callbacks=[earlystop, 
                checkpoint, reduce_lr])
    else:
        model.fit(dataset['train'], epochs=1,validation_data=dataset['val'])
        model.load_weights(tf.train.latest_checkpoint(directory))
    return model
