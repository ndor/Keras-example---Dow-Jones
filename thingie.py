#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import h5py
import csv
import json
import sys
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Model
from keras.layers import Input, Dense, Dropout, concatenate, Flatten, SimpleRNN, GRU, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D

import pprint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pp = pprint.PrettyPrinter()

_general_size_ = 6

def load_data(path2csv='dji.csv'):
    data = []
    titles = []
    with open(path2csv, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i == 0:
                for r in row:
                    titles.append(r)
                pass
            else:
                data.append(row[1:])
            i += 1
    data = np.array(data, dtype='float32')
    print data.dtype, data.shape
    for i, r in enumerate(titles[1:]):
        print i, r, '| mean :', data[:, i].mean(), \
                    '| std :', data[:, i].std(), \
                    '| min :', data[:, i].min(), \
                    '| max :', data[:, i].max()
    return data[:, :-1], data[:, -1]


def sliding_window_dataset(x, y, window_size):
    X = []
    Y = y[window_size:]
    for i in range(len(x) - window_size):
        X.append(x[i:i + window_size])
    X = np.array(X, dtype='float32')
    return X, Y


def FC_net():
    input = Input(shape=(30,))
    fc = Dense(_general_size_, activation='relu')(input)
    fc = Dropout(0.5)(fc)
    output = Dense(1, activation='linear')(fc)

    model = Model(inputs=input, output=output)
    model.compile(optimizer='adam',
                  metrics=['mse'],
                  loss='mse')
    model.summary()
    return model
def FC_train():
    epochs = 10000
    size_batch = 64
    validation = 0.5

    # loading train & validation data:
    x, y = load_data()
    X = x[:len(x) / 2]
    Y = y[:len(y) / 2]
    print Y.min(), Y.max(), Y.mean(), Y.std()

    # loading the net:
    model = FC_net()
    checkpointer = ModelCheckpoint('FC_thingie.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='min')

    history = model.fit(X,
                        Y,
                        batch_size=size_batch,
                        nb_epoch=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_split=validation,
                        shuffle=True)

    h = history.__dict__
    # pp.pprint(h)
    H = {'params': h['params'], 'history': h['history'], 'epoch': h['epoch']}
    with open('FC_thingie.json', 'wb') as fp:
        json.dump(H, fp)


def CONV_net(window_size):
    input = Input(shape=(window_size, 30))
    conv = Conv1D(window_size, 3, activation='relu', padding='same')(input)
    mp = MaxPooling1D(2)(conv)
    ap = AveragePooling1D(2)(conv)
    conv = concatenate([mp, ap])
    conv = Conv1D(window_size, 3, activation='relu', padding='valid')(conv)
    fc = Flatten()(conv)
    fc = Dropout(0.5)(fc)
    output = Dense(1, activation='linear')(fc)

    model = Model(inputs=input, output=output)
    model.compile(optimizer='adam',
                  metrics=['mse'],
                  loss='mse')
    model.summary()
    return model
def CONV_train():
    epochs = 10000
    size_batch = 64
    validation = 0.5
    window_size = 6

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    X = x[:len(x) / 2]
    Y = y[:len(y) / 2]
    # print Y.min(), Y.max(), Y.mean(), Y.std()

    # loading the net:
    model = CONV_net(window_size=_general_size_)
    checkpointer = ModelCheckpoint('CONV_thingie.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='min')

    history = model.fit(X,
                        Y,
                        batch_size=size_batch,
                        nb_epoch=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_split=validation,
                        shuffle=True)

    h = history.__dict__
    # pp.pprint(h)
    H = {'params': h['params'], 'history': h['history'], 'epoch': h['epoch']}
    with open('CONV_thingie.json', 'wb') as fp:
        json.dump(H, fp)


def SRNN_net(window_size):
    input = Input(shape=(window_size, 30))
    fc = SimpleRNN(window_size, activation='linear', dropout=0.1, recurrent_dropout=0.1)(input)
    output = Dense(1, activation='linear')(fc)

    model = Model(inputs=input, output=output)
    model.compile(optimizer='adam',
                  metrics=['mse'],
                  loss='mse')
    model.summary()
    return model
def SRNN_train():
    epochs = 10000
    size_batch = 64
    validation = 0.5
    window_size = 6

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    X = x[:len(x) / 2]
    Y = y[:len(y) / 2]
    # print Y.min(), Y.max(), Y.mean(), Y.std()

    # loading the net:
    model = SRNN_net(window_size=_general_size_)
    checkpointer = ModelCheckpoint('SRNN_thingie.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='min')

    history = model.fit(X,
                        Y,
                        batch_size=size_batch,
                        nb_epoch=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_split=validation,
                        shuffle=True)

    h = history.__dict__
    # pp.pprint(h)
    H = {'params': h['params'], 'history': h['history'], 'epoch': h['epoch']}
    with open('SRNN_thingie.json', 'wb') as fp:
        json.dump(H, fp)


def GRU_net(window_size):
    input = Input(shape=(window_size, 30))
    fc = GRU(window_size, activation='linear', dropout=0.1, recurrent_dropout=0.1)(input)
    output = Dense(1, activation='linear')(fc)

    model = Model(inputs=input, output=output)
    model.compile(optimizer='adam',
                  metrics=['mse'],
                  loss='mse')
    model.summary()
    return model
def GRU_train():
    epochs = 10000
    size_batch = 64
    validation = 0.5
    window_size = 6

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    X = x[:len(x) / 2]
    Y = y[:len(y) / 2]
    # print Y.min(), Y.max(), Y.mean(), Y.std()

    # loading the net:
    model = GRU_net(window_size=_general_size_)
    checkpointer = ModelCheckpoint('GRU_thingie.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='min')

    history = model.fit(X,
                        Y,
                        batch_size=size_batch,
                        nb_epoch=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_split=validation,
                        shuffle=True)

    h = history.__dict__
    # pp.pprint(h)
    H = {'params': h['params'], 'history': h['history'], 'epoch': h['epoch']}
    with open('GRU_thingie.json', 'wb') as fp:
        json.dump(H, fp)


def LSTM_net(window_size):
    input = Input(shape=(window_size, 30))
    fc = LSTM(window_size, activation='linear', dropout=0.1, recurrent_dropout=0.1)(input)
    output = Dense(1, activation='linear')(fc)

    model = Model(inputs=input, output=output)
    model.compile(optimizer='adam',
                  metrics=['mse'],
                  loss='mse')
    model.summary()
    return model
def LSTM_train():
    epochs = 10000
    size_batch = 64
    validation = 0.5
    window_size = 6

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    X = x[:len(x) / 2]
    Y = y[:len(y) / 2]
    # print Y.min(), Y.max(), Y.mean(), Y.std()

    # loading the net:
    model = LSTM_net(window_size=_general_size_)
    checkpointer = ModelCheckpoint('LSTM_thingie.hdf5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='min')

    history = model.fit(X,
                        Y,
                        batch_size=size_batch,
                        nb_epoch=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_split=validation,
                        shuffle=True)

    h = history.__dict__
    # pp.pprint(h)
    H = {'params': h['params'], 'history': h['history'], 'epoch': h['epoch']}
    with open('LSTM_thingie.json', 'wb') as fp:
        json.dump(H, fp)



def FC_display(path2json):
    with open(path2json) as fp:
        J = json.load(fp)
    print J.keys()
    # pp.pprint(J)
    x = J['epoch']
    loss = J['history']['loss']
    val_loss = J['history']['val_loss']
    plt.plot(x, loss, x, val_loss)
    plt.show()

    model = FC_net()
    #######################################
    # if previous file exist:
    F = path2json.split('.')[0] + '.hdf5'
    if os.path.isfile(F):
        print 'loading weights file: ' + F
        model.load_weights(F)
    #######################################

    # loading train & validation data:
    x, y = load_data()
    print len(y)
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]
    Ytag = model.predict_on_batch(X)
    plt.plot(range(len(Y)), Y, range(len(Ytag)), Ytag)
    plt.show()


def CONV_display(path2json):
    with open(path2json) as fp:
        J = json.load(fp)
    print J.keys()
    # pp.pprint(J)
    x = J['epoch']
    loss = J['history']['loss']
    val_loss = J['history']['val_loss']
    plt.plot(x, loss, x, val_loss)
    plt.show()

    window_size = 6
    model = CONV_net(window_size)
    #######################################
    # if previous file exist:
    F = path2json.split('.')[0] + '.hdf5'
    if os.path.isfile(F):
        print 'loading weights file: ' + F
        model.load_weights(F)
    #######################################

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    print len(y)
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]
    Ytag = model.predict_on_batch(X)
    plt.plot(range(len(Y)), Y, range(len(Ytag)), Ytag)
    plt.show()


def SRNN_display(path2json):
    with open(path2json) as fp:
        J = json.load(fp)
    print J.keys()
    # pp.pprint(J)
    x = J['epoch']
    loss = J['history']['loss']
    val_loss = J['history']['val_loss']
    plt.plot(x, loss, x, val_loss)
    plt.show()

    window_size = 6
    model = SRNN_net(window_size)
    #######################################
    # if previous file exist:
    F = path2json.split('.')[0] + '.hdf5'
    if os.path.isfile(F):
        print 'loading weights file: ' + F
        model.load_weights(F)
    #######################################

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    print len(y)
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]
    Ytag = model.predict_on_batch(X)
    plt.plot(range(len(Y)), Y, range(len(Ytag)), Ytag)
    plt.show()


def GRU_display(path2json):
    with open(path2json) as fp:
        J = json.load(fp)
    print J.keys()
    # pp.pprint(J)
    x = J['epoch']
    loss = J['history']['loss']
    val_loss = J['history']['val_loss']
    plt.plot(x, loss, x, val_loss)
    plt.show()

    window_size = 6
    model = GRU_net(window_size)
    #######################################
    # if previous file exist:
    F = path2json.split('.')[0] + '.hdf5'
    if os.path.isfile(F):
        print 'loading weights file: ' + F
        model.load_weights(F)
    #######################################

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    print len(y)
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]
    print X.dtype
    Ytag = model.predict_on_batch(X)
    plt.plot(range(len(Y)), Y, range(len(Ytag)), Ytag)
    plt.show()


def LSTM_display(path2json):
    with open(path2json) as fp:
        J = json.load(fp)
    print J.keys()
    # pp.pprint(J)
    x = J['epoch']
    loss = J['history']['loss']
    val_loss = J['history']['val_loss']
    plt.plot(x, loss, x, val_loss)
    plt.show()

    window_size = 6
    model = LSTM_net(window_size)
    #######################################
    # if previous file exist:
    F = path2json.split('.')[0] + '.hdf5'
    if os.path.isfile(F):
        print 'loading weights file: ' + F
        model.load_weights(F)
    #######################################

    # loading train & validation data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, window_size)
    print len(y)
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]
    print X.dtype
    Ytag = model.predict_on_batch(X)
    plt.plot(range(len(Y)), Y, range(len(Ytag)), Ytag)
    plt.show()


def display_all():

    # load models & history:
    FC_model = FC_net()
    FC_model.load_weights('FC_thingie.hdf5')
    # FC_x = J['epoch']
    # FC_loss = J['history']['loss']
    # FC_val_loss = J['history']['val_loss']

    CONV_model = CONV_net(window_size=_general_size_)
    CONV_model.load_weights('CONV_thingie.hdf5')
    # CONV_x = J['epoch']
    # CONV_loss = J['history']['loss']
    # CONV_val_loss = J['history']['val_loss']

    SRNN_model = SRNN_net(window_size=_general_size_)
    SRNN_model.load_weights('SRNN_thingie.hdf5')
    # SRNN_x = J['epoch']
    # SRNN_loss = J['history']['loss']
    # SRNN_val_loss = J['history']['val_loss']

    GRU_model = GRU_net(window_size=_general_size_)
    GRU_model.load_weights('GRU_thingie.hdf5')
    # GRU_x = J['epoch']
    # GRU_loss = J['history']['loss']
    # GRU_val_loss = J['history']['val_loss']

    LSTM_model = LSTM_net(window_size=_general_size_)
    LSTM_model.load_weights('LSTM_thingie.hdf5')
    # LSTM_x = J['epoch']
    # LSTM_loss = J['history']['loss']
    # LSTM_val_loss = J['history']['val_loss']

    # plain test data:
    x, y = load_data()
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]
    Y_gt = Y.copy()

    # test FC:
    FC_Y = FC_model.predict_on_batch(X)
    FC_eval = FC_model.evaluate(X, Y, batch_size=len(X))

    # sliding window test data:
    x, y = load_data()
    x, y = sliding_window_dataset(x, y, _general_size_)
    X = x[len(x) / 2:]
    Y = y[len(y) / 2:]

    # test CONV:
    CONV_Y = CONV_model.predict_on_batch(X)
    CONV_eval = CONV_model.evaluate(X, Y, batch_size=len(X))

    # test SRNN:
    SRNN_Y = SRNN_model.predict_on_batch(X)
    SRNN_eval = SRNN_model.evaluate(X, Y, batch_size=len(X))

    # test GRU:
    GRU_Y = GRU_model.predict_on_batch(X)
    GRU_eval = GRU_model.evaluate(X, Y, batch_size=len(X))

    # test LSTM:
    LSTM_Y = LSTM_model.predict_on_batch(X)
    LSTM_eval = LSTM_model.evaluate(X, Y, batch_size=len(X))


    # plotting:
    x = range(len(Y))
    plt.plot(x, Y_gt[_general_size_ / 2:], 'r', x, FC_Y[_general_size_ / 2:], 'm', x, CONV_Y, 'b', x, SRNN_Y, 'g', x, GRU_Y, 'k', x, LSTM_Y, 'c')
    plt.xlabel('time [day]')
    plt.ylabel('DJi')
    plt.title('comparing DJi predictions of different NN models')
    plt.grid(True)
    plt.legend(('GT', 'FC', 'CONV', 'SRNN', 'GRU', 'LSTM'))
    plt.show()


    # # plotting:
    # x = range(len(Y))
    # plt.plot( x, np.abs(FC_Y[_general_size_ / 2:] - Y_gt[_general_size_ / 2:]), 'm', x, np.abs(CONV_Y - Y_gt[_general_size_ / 2:]), 'b', x, np.abs(SRNN_Y - Y_gt[_general_size_ / 2:]), 'g', x, np.abs(GRU_Y - Y_gt[_general_size_ / 2:]), 'k', x, np.abs(LSTM_Y - Y_gt[_general_size_ / 2:]), 'c', label='dLSTM')
    # plt.xlabel('time [day]')
    # plt.ylabel('DJi')
    # plt.title('comparing DJi predictions of different NN models')
    # plt.grid(True)
    # plt.legend()
    # plt.show()



    # with open(path2json) as fp:
    #     J = json.load(fp)
    # print J.keys()
    # # pp.pprint(J)
    # x = J['epoch']
    # loss = J['history']['loss']
    # val_loss = J['history']['val_loss']
    # plt.plot(x, loss, x, val_loss)
    # plt.show()
    #
    # window_size = 6
    # model = LSTM_net(window_size)
    # #######################################
    # # if previous file exist:
    # F = path2json.split('.')[0] + '.hdf5'
    # if os.path.isfile(F):
    #     print 'loading weights file: ' + F
    #     model.load_weights(F)
    # #######################################
    #
    # # loading train & validation data:
    # x, y = load_data()
    # x, y = sliding_window_dataset(x, y, window_size)
    # print len(y)
    # X = x[len(x) / 2:]
    # Y = y[len(y) / 2:]
    # print X.dtype
    # Ytag = model.predict_on_batch(X)
    # plt.plot(range(len(Y)), Y, range(len(Ytag)), Ytag)
    # plt.show()




if __name__ == '__main__':
    # FC_train()
    # FC_display('FC_thingie.json')
    # CONV_train()
    # CONV_display('CONV_thingie.json')
    # SRNN_train()
    # SRNN_display('SRNN_thingie.json')
    # GRU_train()
    # GRU_display('GRU_thingie.json')
    # LSTM_train()
    # LSTM_display('LSTM_thingie.json')


    display_all()





















