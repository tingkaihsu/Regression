import tensorflow as tf
from tensorflow import keras

class _DotDict:
    def __init__(self, num_class):
        self.num_class = num_class

def simple_model_base(features=None, setting=None, name='Simple_Model'):
    with tf.name_scope(name):
        out = keras.layers.Dense(setting.num_class, activation='linear')(features)
        return out
    
def get_simple_model(num_classes, input_shapes):
    setting = _DotDict(num_class=num_classes)
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    outputs = simple_model_base(features, setting)
    return keras.Model(inputs=[features], outputs=outputs, name='Simple_Model')

