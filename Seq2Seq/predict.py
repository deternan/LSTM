# -*- coding: utf-8 -*-

'''
Created on February 12, 2020  09:15 AM
revised on February 12, 2020  02:30 PM
'''


'''
Reference
https://keras.io/examples/lstm_seq2seq_restore/
https://github.com/samurainote/seq2seq_translate_slackbot/blob/master/0315_seq2seq_translater.ipynb
'''


from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

data_path = 'D:\\Phelps\\gitlab\\NeuralNetwork\\Seq2Seq\\dataset\\cmn.txt'
saveMode = 'seq2seq.h5'

# data processing
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 15000  # Number of samples to train on.

with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

for line in lines[: min(num_samples, len(lines) - 1)]:
  input_text, target_text, lincene = line.split('\t')
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

#print(input_characters)
#print(target_characters)

input_token_index = dict(
  [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
  [(char, i) for i, char in enumerate(target_characters)])

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs],
              outputs=decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights(saveMode)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
  decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states)

# reverse-lookup token index to turn sequences back to characters
reverse_input_char_index = dict(
  (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
  (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # encode the input sequence to get the internal state vectors.
    states_value = encoder_model.predict(input_seq)

    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    # output sequence loop
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # sample a token and add the corresponding character to the
        # decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # check for the exit condition: either hitting max length
        # or predicting the 'stop' character
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # update the target sequence (length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # update states
        states_value = [h, c]

    return decoded_sentence


# for seq_index in range(10):
#   input_seq = encoder_input_data[seq_index: seq_index + 1]
#   decoded_sentence = decode_sequence(input_seq)
#   print('-')
#   print('Input sentence:', input_texts[seq_index])
#   print('Decoded sentence:', decoded_sentence)

# Input and Display
input_sentence = "How are you?"
test_sentence_tokenized = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
for t, char in enumerate(input_sentence):
    test_sentence_tokenized[0, t, input_token_index[char]] = 1.
print(input_sentence)
print(decode_sequence(test_sentence_tokenized))
