# Copyright © UIF (University Industry Foundation), Yonsei University. All rights reserved. For any inquiries regarding the usage or licensing of this code, please contact the UIF. (Contact: +82-2-2123-5176 / jemin17@yonsei.ac.kr)

import tensorflow as tf
from keras import layers, Model

# Gated linear unit
class GLU(layers.Layer):
    def call(self, inputs):
        a, b = tf.split(inputs, num_or_size_splits=2, axis=-1)
        return a * tf.sigmoid(b)

# DPL
class DPL(Model):
    def __init__(self, nr_units=50, nr_layers=3):
        super(DPL, self).__init__()
        self.act_func = layers.LeakyReLU()
        self.last_act_func = GLU()
        
        self.hidden_layers = []
        self.hidden_layers.append(layers.Dense(nr_units))
        self.hidden_layers.append(self.act_func)
        for _ in range(2, nr_layers + 1):
            self.hidden_layers.append(layers.Dense(nr_units))
            self.hidden_layers.append(self.act_func)
        self.last_layer = layers.Dense(3)
    
    def call(self, inputs):
        '''
        Args:
            inputs: np.ndarray
                Sensor parameter configuration and its scaled cost. (num_samples, num_features+1)
        '''
        x = inputs[:,:-1]
        cost = inputs[:,-1]
        
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.last_layer(x)

        alphas = x[:, 0]
        betas = x[:, 1]
        gammas = x[:, 2]

        betas_cat = tf.stack([betas, betas], axis=-1)
        gammas_cat = tf.stack([gammas, gammas], axis=-1)

        betas_glu = self.last_act_func(betas_cat)
        gammas_glu = self.last_act_func(gammas_cat)

        cost = tf.reshape(cost, (-1,1)) # Budget
        power = tf.math.pow(cost, -gammas_glu)
        power = tf.squeeze(power, axis=1)
        mul = tf.squeeze(betas_glu, axis=1) * power
        output = alphas + mul

        return output
