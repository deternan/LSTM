# coding=utf8

"""
Created on Fri December 13 2019, 05:12 PM

@author: BARRY.KE
"""

'''
Reference

https://medium.com/datainpoint/numpy102-a052665ccf44

'''

import tensorflow as tf

#from keras.models import Model
#from keras.models import load_model
#from keras.layers import Input, LSTM, Dense
import numpy as np
#import tensorflow.keras.optimizers

print('tensorflow version ', tf.__version__)

# tensorflow
a = tf.constant([1,2,3])
a = np.array([1,2,3])
b = tf.convert_to_tensor(a)
#print(a)
#print(b)

#Example create zero
a = tf.zeros([6,6])
#or
b = tf.zeros_like(a)
#or fill -> shape and fill nums
c = tf.fill([6,6],0)
#print(a)
#print(b)
#print(c)

#Example random by normal distribution
a = tf.random.normal([6,6],mean=0,stddev=1)
#random by truncated normal distribution
b = tf.random.truncated_normal([6,6],mean=0,stddev=1)
#random by uniform
c = tf.random.uniform([6,6],minval=0,maxval=1)
#print(a)
#print(b)
#print(c)


a = tf.random.normal([6,32,32,3])
#Reshape to different type
tf.reshape(a,[6,32*32,3]).shape
#print(a)

# show the value
zero_d = np.array(5566)            # 零維陣列，純量
one_d = np.array([55, 66, 5566])   # 一維陣列
two_d = np.ones((3, 3), dtype=int) # 二維陣列
#print("ndim:")
#print(zero_d.ndim)
#print(one_d.ndim)
#print(two_d.ndim)
#print("shape:")
#print(zero_d.shape)
#print(one_d.shape)
#print(two_d.shape)
#print("size:")
#print(zero_d.size)
#print(one_d.size)
#print(two_d.size)
#print("dtype:")
#print(zero_d.dtype)
#print(one_d.dtype)
#print(two_d.dtype)


# sort
data = tf.random.normal([3,3], mean=0, stddev=1)
tf.sort(data, direction='DESCENDING')

#print(data.shape)

#or
tf.gather(data,tf.argsort(data,direction='ASCENDING'))

a = tf.fill([2,2],5)
#print(a)

n_samples = 10
#Create dataset
X = np.random.rand(n_samples).astype(np.float32)
Y = X * 10 + 5
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

#print(X)
#print(Y)
#print(W)
#print(b)


#Define LR and Loss function (MSE)
def linear_regression(x):
    return W * x + b
  
def mean_square(y_pred, y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true, 2)) / (2 * n_samples)

learning_rate = 0.1
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



