from tox21_models.utils import create_model
from kerastuner import HyperParameters
import tensorflow.keras as tfk
import tensorflow as tf
def study_buildclass():
    hp = HyperParameters()
    hp.Fixed('learning_rate', value=1e-3)
    hp.Fixed('concat_at', value=1)
    hp.Fixed('units_0', value=400)
    hp.Fixed('units_1', value=400)
    hp.Fixed('units_2', value=400)

    hp.Fixed('dropout_0', value=0.1)
    hp.Fixed('dropout_1', value=0.1)
    hp.Fixed('dropout_2', value=0.1)
    sex = hp.values
    print (type(sex))

study_buildclass()
