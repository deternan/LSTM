
'''
version: January 03 2020, 02:40 PM
Last revision: January 03 2020, 02:40 PM

Author : Chao-Hsuan Ke
'''

from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import LSTM, Dense

# define problem properties
# stpes
n_timesteps = 10
# value number
maxNum = 10

# create a sequence of random numbers in [0,maxNum]
X = array([random() for _ in range(maxNum)])
# calculate cut-off value to change class values
limit = maxNum/4.0


# determine the class outcome for each item in cumulative sequence
Y = array([0 if x < limit else 1 for x in cumsum(X)])

# create a sequence classification instance
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = array([random() for _ in range(n_timesteps)])
    # calculate cut-off value to change class values
    limit = n_timesteps/4.0
    # determine the class outcome for each item in cumulative sequence
    y = array([0 if x < limit else 1 for x in cumsum(X)])
    X = X.reshape(1, n_timesteps, 1)
    y = y.reshape(1, n_timesteps, 1)
    return X, y

X, Y = get_sequence(n_timesteps)
# print(X)
# print(Y)


# reshape input and output data to be suitable for LSTMs
X = X.reshape(1, n_timesteps, 1)
Y = Y.reshape(1, n_timesteps, 1)

# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# train LSTM
for epoch in range(1000):
    # generate new random sequence
    X,Y = get_sequence(n_timesteps)
    # fit model for one epoch on this sequence
    model.fit(X, Y, epochs=1, batch_size=1, verbose=2)

#print(X)
#print(Y)

# evaluate LSTM
X,Y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('Expected:', Y[0, i], 'Predicted', yhat[0, i])