
'''
version: January 03 2020, 03:40 PM
Last revision: January 03 2020, 03:42 PM

Author : Chao-Hsuan Ke
'''

'''
Reference
https://www.jianshu.com/p/f12dc3245ab8
'''


from random import random
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# steps
n_timesteps = 10
# value number
maxNum = 10

# create a sequence classification instance
def get_sequence(n_timesteps):
    # create a sequence of random numbers in [0,1]
    X = array([random() for _ in range(n_timesteps)])
    # calculate cut-off value to change class values
    limit = maxNum/4.0
    # determine the class outcome for each item in cumulative sequence
    Y = array([0 if x < limit else 1 for x in cumsum(X)])
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)
    Y = Y.reshape(1, n_timesteps, 1)
    return X, Y

def get_lstm_model(n_timesteps, backwards):
    model = Sequential()
    model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True, go_backwards=backwards))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def get_bi_lstm_model(n_timesteps, mode):
    model = Sequential()
    model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1), merge_mode=mode))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_model(model, n_timesteps):
    loss = list()
    for _ in range(250):
        # generate new random sequence
        X,Y = get_sequence(n_timesteps)
        # fit model for one epoch on this sequence
        hist = model.fit(X, Y, epochs=1, batch_size=1, verbose=0)
        loss.append(hist.history['loss'][0])
    return loss


n_timesteps = 10
results = DataFrame()
# lstm forwards
model = get_lstm_model(n_timesteps, False)
results['lstm_forw'] = train_model(model, n_timesteps)
# lstm backwards
model = get_lstm_model(n_timesteps, True)
results['lstm_back'] = train_model(model, n_timesteps)
# bidirectional concat
model = get_bi_lstm_model(n_timesteps, 'concat')
results['bilstm_con'] = train_model(model, n_timesteps)
# line plot of results
results.plot()
pyplot.show()