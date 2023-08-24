import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shekespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data'
                                                      '/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

model = tf.keras.models.load_model('textgenerator.model')

def sample(preds, temperature=0.1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    strat_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sent = text[strat_index:strat_index + SEQ_LENGTH]
    sentence = sent
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)), dtype=np.bool_)
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predections = model.predict(x, verbose=0)[0]
        next_index = sample(predections, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character

    return sent + generated

print(generate_text(300, 0.5))

