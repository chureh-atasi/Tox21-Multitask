import os
from kerastuner import HyperParameters
import tensorflow.keras as tfk
import tensorflow as tf
from tox21_models.utils.models import (build_regression_model,  
        build_classification_model
)

def build_classification(dataset, save_folder, build_new: bool = True):
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
    model = build_classification_model(hp)
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
