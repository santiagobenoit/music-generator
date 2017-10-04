import datetime
import glob
import keras
from keras.layers import *
from keras.models import *
import numpy
import os
import random
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
    for chord in range(1, chords):
        low_candidate = numpy.argmax(sequence[chord, :])
        high_candidate = notes - numpy.argmax(sequence[chord, ::-1])
        if low_candidate < low:
            low = low_candidate
        if high_candidate > high:
            high = high_candidate
    return low, high


def augment(sequence):
    chords = sequence.shape[0]
    low, high = low_high(sequence)
    transpositions = notes - high + 1
    data = []
    for i in range(transpositions):
        progression = numpy.empty((chords, notes))
        for chord in range(chords):
            progression[chord, :] = numpy.concatenate((
                numpy.zeros((i,)),
                sequence[chord, low:high],
                numpy.zeros((notes - (high - low) - i,))
            ))
        data.append(progression)
    return data


def generator(data_dir, seed=None):
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if seed is not None:
        random.seed(seed)
    while True:
        for data_file in data_files:
            sequences = []
            data = numpy.load(data_file)
            for key in data:
                sequences.append(data[key])
            augmented = []
            for sequence in sequences:
                sequence = sequence[0, :, :]
                sequence = numpy.insert(sequence, 0, numpy.zeros((notes,)), axis=0)
                augmented.extend(augment(sequence))
            random.shuffle(augmented)
            for sequence in augmented:
                yield sequence[numpy.newaxis, :, :], numpy.vstack([sequence[1:, :], sequence[1, :]])[numpy.newaxis, :, :]


def init_model(submodel):
    model = None
    if submodel == 'chords':
        input_layer = Input((None, notes))
        hidden_layer = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2), merge_mode='concat')(input_layer)
        output_layer = TimeDistributed(Dense(notes, activation='sigmoid'))(hidden_layer)
        model = Model(input_layer, output_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def train(submodel, data_dir, steps_per_epoch, epochs, callbacks=None, seed=None):
    model = init_model(submodel)
    model.summary()
    model.fit_generator(generator(data_dir, seed), steps_per_epoch, epochs=epochs, callbacks=callbacks)
    return model


def evaluate(model, data_dir, steps, seed=None):
    return model.evaluate_generator(generator(data_dir, seed), steps)


def predict(model, x):
    return model.predict(x)
