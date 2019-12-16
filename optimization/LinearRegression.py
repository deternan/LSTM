# coding=utf8

'''
version: December 16, 2019 10:45 AM
Last revision: December 16, 2019 01:14 PM

Author : Chao-Hsuan Ke
'''

import tensorflow as tf

#from keras.models import Model
#from keras.models import load_model
#from keras.layers import Input, LSTM, Dense
import numpy as np
#import tensorflow.keras.optimizers

n_samples = 10
#Create dataset
X = np.random.rand(n_samples).astype(np.float32)
Y = X * 10 + 5
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))


# Define LR and Loss function (MSE)
def linear_regression(x):
    return W * x + b

# evaluate the error (lose function)
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred - y_true, 2)) / (2 * n_samples)


learning_rate = 0.5
# Stochastic Gradient Descent Optimizer.
#optimizer = tf.optimizers.SGD(learning_rate)
#optimizer (Adam)
optimizer = tf.optimizers.Adam(learning_rate)

def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))


training_steps = 1000
display_step = 1

for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()
    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))






