"""Models taken from kuelumbus' work"""
import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.engine import data_adapter

class PropertyDownstreamClassifier(tfk.Model):
    """Classifier model"""
    def __init__(self, hp):
        super().__init__()
        self.my_layers = []
        # name = concat_at, 0 = min, 2 = max
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
        # Can classify as either low, medium, or high
        self.my_layers.append([tf.keras.layers.Dense(3)])
        # Add classification capability
        classifier_step = [
            tf.keras.layers.Dense(3, activation='softmax')  
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
    opt = tf.keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-3]))
    opt = tfa.optimizers.SWA(opt)

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),)
    return model
