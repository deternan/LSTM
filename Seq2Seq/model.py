# -*- coding: utf-8 -*-

'''
Created on February 11, 2020  05:25 PM
revised on February 11, 2020  06:30 PM
'''


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

data_path = 'D:\\Phelps\\gitlab\\NeuralNetwork\\Seq2Seq\\dataset\\cmn.txt'
saveMode = 'seq2seq.h5'

## data processing
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 15000  # Number of samples to train on.

# Path to the data txt file on disk.
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, lincene = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


## Encoder-LSTM
# mapping token to index，easily to vectors
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# np.zeros(shape, dtype, order)
# shape is an tuple, in here 3D
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# input_texts contain all english sentences
# output_texts contain all chinese sentences
# zip('ABC','xyz') ==> Ax By Cz, looks like that
# the aim is: vectorilize text, 3D
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        # 3D vector only z-index has char its value equals 1.0
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            # igone t=0 and start t=1, means
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


## Context （ hidden state ）

# Define an input sequence and process it.
# input prodocts keras tensor, to fit keras model!
# 1x73 vector
# encoder_inputs is a 1x73 tensor!
encoder_inputs = Input(shape=(None, num_encoder_tokens))

# units=256, return the last state in addition to the output
encoder_lstm = LSTM((latent_dim), return_state=True)

# LSTM(tensor) return output, state-history, state-current
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


## Decoder-LSTM
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM((latent_dim), return_sequences=True, return_state=True)

# obtain output
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

# dense 2580x1 units full connented layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

# why let decoder_outputs go through dense ?
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn, groups layers into an object
# with training and inference features
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# model(input, output)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Run training
# compile -> configure model for training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model optimization (1)
# model.fit([encoder_input_data, decoder_input_data],
#           decoder_target_data,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2)

# model optimization (2)
model.fit([np.array(encoder_input_data), np.array(decoder_input_data)],
          np.array(decoder_target_data),
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
model.save(saveMode)

print('finished')
