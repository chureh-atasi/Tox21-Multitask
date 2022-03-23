"""Run to test new multitask models"""
import os
import copy
import pandas as pd
from keras_tuner import HyperParameters
import tensorflow.keras as tfk
import tensorflow as tf
from tox21_models import interface_binary_multi
from tox21_models.utils.data_loader import create_datasets
from tox21_models.utils import data_loader, tuner_multi
from tox21_models.utils.models_multi import MultiTaskModel

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

save_folder = 'example'

"""
df = pd.read_csv(os.path.join('..', 'tox21_models', 'data', 'temp.csv'))
#df = df.loc[df.prop == 'Liver']
props = list(df.prop.unique())
fp_headers = [col for col in df.columns if 'fp_' in col]
datasets = create_datasets(df, list(df.prop.unique()), fp_headers, False)

hps = {}
for prop in ['input'] + props:
    thp = HyperParameters()
    thp.Fixed('learning_rate', value=1e-4)
    thp.Fixed('units_0', value=100)
    thp.Fixed('units_1', value=100)
    thp.Fixed('units_2', value=100)
    thp.Fixed('num_layers', value=3)

    thp.Fixed('dropout_0', value=0)
    thp.Fixed('dropout_1', value=0)
    thp.Fixed('dropout_2', value=0)
    hps[prop] = copy.deepcopy(thp)

# Create an instance of the model
hypermodel = MultiTaskModel(2)
model_hp = HyperParameters()
model = hypermodel.build(hp=model_hp, hps=hps, keys=props) 
directory = os.path.join(save_folder, 'chkpt')
for num, data in enumerate(datasets):
    train = data[0]['train']
    val = data[0]['val']

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
            patience=60)

    hist = model.fit(train, epochs=300,
            validation_data=val, callbacks=[earlystop, 
            checkpoint, reduce_lr])

    break
"""

interface_binary_multi.run(['Liver', 'Hamster'], 'example', False)
