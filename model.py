import glob
import keras
from keras.layers import *
from keras.models import *
import numpy
import os
import tensorflow

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)
keras.backend.set_session(session)

notes = 24

def low_high(sequence):
    chords = sequence.shape[0]
    low = notes - 1
    high = 0
    for chord in range(chords):
        low_candidate = numpy.argmax(sequence[chord, :])
        high_candidate = notes - numpy.argmax(sequence[chord, ::-1])
        if low_candidate < low:
            low = low_candidate
        if high_candidate > high:
            high = high_candidate
    return low, high

def generate(sequence):
    chords = sequence.shape[0]
    low, high = low_high(sequence)
    transpositions = notes - high + 1
    data = numpy.empty((transpositions, chords, notes))
    for i in range(transpositions):
        for chord in range(chords):
            data[i, chord, :] = numpy.concatenate((
                numpy.zeros((i,)),
                sequence[chord, low:high],
                numpy.zeros((notes - (high - low) - i,))
            ))
    return data, numpy.roll(data, -1, axis=1)

def generator(data_dir):
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    for data_file in data_files:
        data = numpy.load(data_file)
        progressions = data.shape[0]
        for progression in progressions:
            yield generate(data[progression, :, :])

def init_model():
    input_layer = Input((None, notes))
    hidden_layer = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(input_layer)
    output_layer = TimeDistributed(Dense(notes, activation='sigmoid'))(hidden_layer)
    model = Model(input_layer, output_layer)
    return model

def train(data_dir, steps_per_epoch, epochs, callbacks):
    model = init_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    model.fit_generator(generator(data_dir), steps_per_epoch, epochs=epochs, callbacks=callbacks)
