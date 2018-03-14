# -*- coding: utf-8 -*-
"""
Created on Mar 14 2018

a simple practice about linear regression implementing by tensorflow.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_fc_layer(name, inputs, num_input, num_output, activation_func=None):
    """
    Add a full connect layer to Neural Netwoek.
    """
    weight_init = tf.keras.initializers.lecun_normal()
    bias_init = tf.constant_initializer(0.0, tf.float32)

    with tf.name_scope(name):
        weight = tf.get_variable('weight', 
                                 shape=[num_input, num_output],
                                 dtype=tf.float32,
                                 initializer=weight_init)
        bias = tf.get_variable('bias',
                                shape=[num_output],
                                dtype=tf.float32,
                                initializer=bias_init)
        neuron = tf.matmul(inputs, weight) + bias

    if activation_func != None:
        neuron = activation_func(neuron)
    
    return neuron

def main():
    # Make some data for regression
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # reshape to (1, 300)
    noise = np.random.normal(0.0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # Define placeholder for network input
    x = tf.placeholder(tf.float32, [None, 1], 'input')
    y = tf.placeholder(tf.float32, [None, 1], 'label')

    fc_1 = add_fc_layer('fc_1', x, 1, 10, tf.nn.relu)
    output = add_fc_layer('output', fc_1, 10, 1, None)