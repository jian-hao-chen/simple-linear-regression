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

    with tf.variable_scope(name):
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
    tf.reset_default_graph()
    # make some data for regression
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # reshape to (1, 300)
    noise = np.random.normal(0.0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise

    # define placeholder for network input
    x = tf.placeholder(tf.float32, [None, 1], 'input')
    y = tf.placeholder(tf.float32, [None, 1], 'label')

    # add two full connected layer
    fc_1 = add_fc_layer('fc_1', x, 1, 10, tf.nn.relu)
    output = add_fc_layer('output', fc_1, 10, 1, None)

    # define loss function: mse
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - output), axis=1))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # prepare figure and draw
        fig = plt.figure()
        subplot = fig.add_subplot(1, 1, 1) # 1 row, 1 colum, 1st image
        subplot.scatter(x_data, y_data)
        plt.ion() # turn interactive mode on
        plt.show()
        lines = [] # not neccesary, it's for hiding the warning message.

        for i in range(1000):
            # training model
            sess.run(train, feed_dict = { x : x_data, y : y_data })

            # validation
            if i % 50 == 0:
                try:
                    subplot.lines.remove(lines[0])
                except Exception:
                    pass
                predict = output.eval(feed_dict = { x : x_data })
                lines = subplot.plot(x_data, predict, 'r-', lw = 2)
                plt.pause(0.1) # To realtime show the result
    plt.ioff()
    # turn interactive mode off, or the figure will be closed after
    # the program
    plt.show()
    print("Compelete.")

    return

if __name__ == '__main__':
    main()