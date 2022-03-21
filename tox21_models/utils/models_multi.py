"""Models taken from kuelumbus' work"""
from typing_extensions import Self
import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.engine import data_adapter
import tensorflow_recommenders as tfrs

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
