"""Models taken from kuelumbus' work"""
from typing import Dict
from typing_extensions import Self
import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.engine import data_adapter
import tensorflow_recommenders as tfrs
from keras_tuner import HyperParameters, HyperModel

class InputNetwork(tfk.Model):
    """Model for learning fingerprint data.
    
    Can be any fingerprint data
    """
    def __init__(self, hp, key):
        super().__init__()
        self.my_layers = []
        self.key = key

        for i in range(hp.Int('num_layers', min_value=1, max_value=3, step=1)): 
            new_step = [               
            tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=50,
                                            max_value=300,
                                            step=50),),
            
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(hp.Float(
                'dropout_' + str(i),
                min_value=0.0,
                max_value=0.5,
                default=0.2,
                step=0.1,
            )),
            ]

            self.my_layers.append(new_step)

    def call(self, inputs, training=False):
        """See manage_data.create_tf_dataset"""
        x = inputs[self.key]

        for num, layer_step in enumerate(self.my_layers):
            for layer in layer_step:
                x = layer(x)
    
        return x

class OutputNetwork(tfk.Model):
    """Takes any input and outputs classification."""

    def __init__(self, hp, num_classes):
        super().__init__()
        self.my_layers = []
        self.num_classes = num_classes

        for i in range(hp.Int('num_layers', min_value=1, max_value=3, step=1)): 
            new_step = [               
            tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=50,
                                            max_value=300,
                                            step=50),),
            
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(hp.Float(
                'dropout_' + str(i),
                min_value=0.0,
                max_value=0.5,
                default=0.2,
                step=0.1,
            )),
            ]

            self.my_layers.append(new_step)

        final_step = [
            tf.keras.layers.Dense(num_classes),
        ]
        self.my_layers.append(final_step)

    def call(self, inputs, training=False):
        """See manage_data.create_tf_dataset"""
        # If there is a type error, x data was passed
        x = inputs

        for num, layer_step in enumerate(self.my_layers):
            for layer in layer_step:
                x = layer(x)

        if self.num_classes == 1: 
            x = tf.keras.activations.sigmoid(x)
        else:
            x = tf.keras.activations.softmax(x) 
        return x
    
    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self(x, training=False)

class MultiTask(tfk.Model):
    """Takes any input and outputs classification."""

    def __init__(self, hps: Dict[str, HyperParameters], keys: Dict[str, str],
            num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.solvent = InputNetwork(hps['solvent'], keys['solvent'])
        self.polymer = InputNetwork(hps['polymer'], keys['polymer'])
        self.concat = OutputNetwork(hps['concat'], num_classes=self.num_classes)


    def call(self, inputs, training=False):
        """See manage_data.create_tf_dataset"""
        sol_x = self.solvent(inputs)
        pol_x = self.polymer(inputs)
        x = tf.concat([sol_x, pol_x], 1)
        x = self.concat(x)
        return x
    
    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self(x, training=False)

class OGSolNetModel(HyperModel):
    """HyperModel with number of classes defined"""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        if num_classes == 2:
            self.num_classes = 1
            self.loss = tf.keras.losses.BinaryCrossentropy()

    def build(self, hp: HyperParameters,
            hps: Dict[str, HyperParameters], keys: Dict[str, str]):
        """Builds output model according to passed hyperparameters

        Args:
            hp (HyperParameters):  
                Hyperparameters for this model.  
            hps (Dict[str, HyperParameters]):  
                Optimizes hyperparameters automatically for polymer, solvent,
                and concat networks. Keys are 'polymer', 'solvent' and 
                'concat'.  
            keys (Dict[str, str]):  
                Key to indicate polymer and solvent data. Dict keys are 
                'polymer' and 'solvent'.

        Returns:
            Compiled model
        """
        model = OGSolNet(hps, keys, self.num_classes)
        opt = tf.keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-4]))
        opt = tfa.optimizers.SWA(opt)

        model.compile(
            optimizer=opt,
            loss=self.loss,)
        return model

class PropertyDownstreamClassifier(tfk.Model):
    """Classifier model"""
    def __init__(self, hp):
        super().__init__()
        self.liver_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.Dense(2000)
        ])
        self.kidney_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.Dense(2000)
        ])
        self.my_layers = []
        # name = concat_at, 0 = min, 2 = max
        self.concat_at = hp.Int('concat_at', 0, 2)
        
        for i in range(hp.Int('num_layers', 1, 3)): 
            new_step = [               
                tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=50,
                                                max_value=200,
                                                step=25),),
                
                tf.keras.layers.PReLU(),
                tf.keras.layers.Dropout(hp.Float(
                    'dropout_' + str(i),
                    min_value=0.0,
                    max_value=0.5,
                    default=0.2,
                    step=0.1,
                )),
            ]

            self.my_layers.append(new_step)
        # Can classify as either low, medium, or high
        # Changed from 3 to 2
        self.kidney_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=tfrs.metrics.BinaryCrossentropy()
        )
        self.liver_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=tfrs.metrics.BinaryCrossentropy()
        )

        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight



        liver_layer_weights = tf.Variable([hp.Int('units_3', min_value=50,
                                                max_value=200,
                                                step=25),2], name="share_Y1")
        kindey_layer_weights = tf.Variable([hp.Int('units_3', min_value=50,
                                                max_value=200,
                                                step=25),1], name="share_Y2")
        self.my_layers.append([tf.keras.layers.Dense(2)])
        # Add classification capability
        # Changed from 3 to 2
        # change from softmax to sigmoid
        # Add the flatten function()
        classifier_step_chicken = [
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Flatten()
        ]
        classifier_step_hamster = [
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Flatten()
        ]
        classifier_step_liver = [
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Flatten()
        ]
        classifier_step_kidney = [
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Flatten()
        ]
        classifier_step_apoptosis = [
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Flatten()
        ]
        list_out = [classifier_step_chicken, classifier_step_hamster, classifier_step_kidney,
                        classifier_step_apoptosis, classifier_step_liver]
        self.my_layers.append(list_out)



    def classification_step(self):
        classifier_step = [
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Flatten()
        ]

        self.my_layers.append(classifier_step)

    def call(self, inputs):
        x = inputs['fps']

        # Model learns best position to add the selector inputs
        for num, layer_step in enumerate(self.my_layers):
            if self.concat_at == num:
                x = tf.concat((x, inputs['selector']), -1)

            for layer in layer_step:
                x = layer(x)
        #Y1_layer = tf.nn.relu(tf.matmul(shared_layer,Y1_layer_weights))
        #Y2_layer_weights = tf.nn.relu(tf.matmul(shared_layer,Y2_layer_weights))
        return x
    
    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        # drop prop here
        prop = x['prop']
        del x['prop']
        return self(x, training=False), data[-1], prop

class PropertyDownstreamRegrssor(tfk.Model):
    """Regression model"""
    def __init__(self, hp):
        super().__init__()
        self.my_layers = []
        self.concat_at = hp.Int('concat_at', 0, 2)
        
        for i in range(hp.Int('num_layers', 3, 3)): 
            new_step = [               
            tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=50,
                                            max_value=300,
                                            step=50),),
            
            tf.keras.layers.PReLU(),
            tf.keras.layers.Dropout(hp.Float(
                'dropout_' + str(i),
                min_value=0.0,
                max_value=0.5,
                default=0.2,
                step=0.1,
            )),
            ]

            self.my_layers.append(new_step)
        self.my_layers.append([tf.keras.layers.Dense(1)])

    def call(self, inputs):
        x = inputs['fps']

        for num, layer_step in enumerate(self.my_layers):
            if self.concat_at == num:
                x = tf.concat((x, inputs['selector']), -1)

            for layer in layer_step:
                x = layer(x)
    
        return x
    
    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        # drop prop here
        prop = x['prop']
        del x['prop']
        return self(x, training=False), data[-1], prop

    
def build_regression_model(hp):
    """Builds regression model according to passed hyperparameters

    Args:
        hp (kerastuner.HyperParameters):
            Optimizes hyperparameters automatically

    Returns:
        Compiled model
    """
    model = PropertyDownstreamRegrssor(hp)
    opt = tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-3]))
    opt = tfa.optimizers.SWA(opt)

    model.compile(
        optimizer=opt,
        loss='mse',)
    return model

def build_classification_model(hp):
    """Builds classification model according to passed hyperparameters

    Args:
        hp (kerastuner.HyperParameters):
            Optimizes hyperparameters automatically

    Returns:
        Compiled model
    """
    model = PropertyDownstreamClassifier(hp)
    model.classification_step(model)
    #model.add(tf.keras.layers.Flatten())
    opt = tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-3]))
    opt = tfa.optimizers.SWA(opt)
    # change from CategoricalCrossentropy to BinaryCrossentropy
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.BinaryCrossentropy(),)
    return model
