# coding=utf8

'''
version: December 16 2019, 01:15 PM
Last revision: December 12, 2019 03:15 PM

Author : Chao-Hsuan Ke
'''


'''
Reference
https://keras.io/zh/datasets/#fashion-mnist\
'''

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras import optimizers

#from tf.keras.models import Model
#from tf.keras.models import load_model
#from tf.keras.layers import Input, LSTM, Dense

#import mnist_reader
#from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt


#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Airbnb New York open dataset
#dataPath = 'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\DNN\\data\\AB_NYC_2019.csv'

# ny_ab = pd.read_csv(dataPath)
# ny_ab.head()
#
# # preprocessing
# ny_ab.drop(['host_name','name','latitude','longitude','last_review','id','host_id'], axis=1, inplace=True)
# ny_ab['reviews_per_month'] = ny_ab['reviews_per_month'].fillna(0)
#
# categorical_features = ny_ab.select_dtypes(include=['object'])
# categorical_features_one_hot = pd.get_dummies(categorical_features)
# categorical_features_one_hot.head()
#
#
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(ny_ab[ny_ab.columns[ny_ab.columns.str.contains('price')==False]])
#ny_ab[ny_ab.columns[ny_ab.columns.str.contains('price')==False]] = x_scaled

# X_train, X_test, y_train, y_test = train_test_split(ny_ab[ny_ab.columns[ny_ab.columns.str.contains('price')==False]] , ny_ab['price'] , test_size=0.1, random_state=66)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(256,activation=tf.nn.relu),
#     tf.keras.layers.Dense(128,activation=tf.nn.relu),
#     tf.keras.layers.Dense(64,activation=tf.nn.relu),
#     tf.keras.layers.Dense(32,activation=tf.nn.relu),
#     tf.keras.layers.Dense(16,activation=tf.nn.relu),
#     tf.keras.layers.Dense(1,activation=tf.nn.relu)
# ])
# model.compile(optimizer='adam',
#                 loss='mean_squared_error',
#                 metrics=['mean_squared_error'])
#
# history = model.fit(X_train.values ,y_train.values, epochs=100, validation_split = 0.1)

# Lab: Fashion-MNIST
(x,y),(x_test,y_test) = fashion_mnist.load_data()
print(x.shape,y.shape)

# graph
# plt.figure()
# plt.imshow(x[0])
# plt.colorbar()
# plt.grid(False)

data = tf.data.Dataset.from_tensor_slices((x,y))

def feature_scale(x,y):
  x = tf.cast(x,dtype=tf.float32)/255.
  y = tf.cast(y,dtype=tf.int32)
  return x,y

data = data.map(feature_scale).shuffle(10000).batch(128)

data_iter = iter(data)
samples = next(data_iter)
print(samples[0].shape,samples[1].shape)


# model
model = Sequential([
     Dense(256,activation=tf.nn.relu),
     Dense(128,activation=tf.nn.relu),
     Dense(64,activation=tf.nn.relu),
     Dense(32,activation=tf.nn.relu),
     Dense(10,activation=tf.nn.relu)
 ])

model.build(input_shape=[None,28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)


# Train
with tf.GradientTape() as tape:
    logits = model(x)
    y_one_hot = tf.one_hot(y, depth=10)
    loss = tf.losses.categorical_crossentropy(y_one_hot,logits,from_logits=True)
    loss = tf.reduce_mean(loss)
grads = tape.gradient(loss,model.trainable_variables)
optimizer.apply_gradients(zip(grads,model.trainable_variables))

# Test
total_loss = 0
x = tf.reshape(x, [-1, 28 * 28])
gd = model(x)
prob = tf.nn.softmax(gd, axis=1)
pred = tf.argmax(prob, axis=1)
pred = tf.cast(pred, dtype=tf.int32)
correct = tf.equal(pred, y)
result = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
total_loss += int(result)

print('loss value', total_loss)

