# -*- coding: utf-8 -*-
# @Time     :1/5/19 10:09 PM
# @Auther   :Jason Lin
# @File     :mismatches_only_test.py
# @Software :PyCharm


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import numpy as np
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LSTM, BatchNormalization, Bidirectional, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Model, model_from_json
import encode_data
import os
from sklearn import metrics
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt
from numpy.random import seed
seed(6)
from tensorflow import set_random_seed
set_random_seed(66)
import pickle as pkl
import tensorflow as tf

def load_cd33_data(load_all=False):

    data, reg_labels = encode_data.encode_CD33_mut()
    my_codes = data.reshape(len(data), 1, 23, 4)

    if load_all:
        X_train = my_codes
        y_train = reg_labels
        return X_train, y_train
    else:
    # print(my_codes)
        X_train, X_test, y_train, y_test = train_test_split(my_codes, reg_labels, test_size = 0.2, random_state = 1)
        print(X_train.shape)
        return X_train, X_test, y_train, y_test

def load_ele_guideseq_data():
    data, reg_vals = encode_data.encode_ele_guideseq_data()
    X_train = data.reshape(len(data), 1, 23, 4)
    y_train = reg_vals
    return X_train, y_train


def load_hmg_data():
    data, reg_vals = encode_data.encode_ele_hmg_data()
    X_train = data.reshape(len(data), 1, 23, 4)
    y_train = reg_vals
    return X_train, y_train

def ConvLSTM():
    # X_train, X_test, y_train, y_test = load_cd33_data()

    # X_train, y_train = load_cd33_data(load_all=True)
    X_train_1, y_train_1 = load_cd33_data(load_all=True)
    X_train_2, y_train_2 = load_hmg_data()
    X_train_3, y_train_3 = load_ele_guideseq_data()
    X_train = np.concatenate((X_train_1, X_train_2, X_train_3))
    y_train = np.concatenate((y_train_1, y_train_2, y_train_3))


    inputs = Input(shape=(1, 23, 4), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding="same", activation='relu')(inputs)
    conv_2 = Conv2D(10, (1, 2), padding="same", activation='relu')(inputs)
    conv_3 = Conv2D(10, (1, 3), padding="same", activation='relu')(inputs)
    conv_4 = Conv2D(10, (1, 5), padding="same", activation='relu')(inputs)

    bn_1 = BatchNormalization()(conv_1)
    bn_2 = BatchNormalization()(conv_2)
    bn_3 = BatchNormalization()(conv_3)
    bn_4 = BatchNormalization()(conv_4)

    bn_1 = Reshape((23, 10))(bn_1)
    bn_2 = Reshape((23, 10))(bn_2)
    bn_3 = Reshape((23, 10))(bn_3)
    bn_4 = Reshape((23, 10))(bn_4)

    # BiLSTM
    blstm_1 = Bidirectional(LSTM(4, return_sequences=True, input_shape=(23, 10)))(bn_1)
    blstm_2 = Bidirectional(LSTM(4, return_sequences=True, input_shape=(23, 10)))(bn_2)
    blstm_3 = Bidirectional(LSTM(4, return_sequences=True, input_shape=(23, 10)))(bn_3)
    blstm_4 = Bidirectional(LSTM(4, return_sequences=True, input_shape=(23, 10)))(bn_4)

    bn_output =  keras.layers.concatenate([blstm_1, blstm_2, blstm_3, blstm_4])

    flatten_output = Flatten()(bn_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(20, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.45)(x)
    prediction = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs, prediction)

    adam_opt = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam_opt)
    print(model.summary())
    model.fit(X_train, y_train, batch_size=100, epochs=200, shuffle=True)

    model_path = "./saved_model"
    if os.path.isdir(model_path):
        pass
    else:
        os.mkdir(model_path)

    model_jason = model.to_json()
    with open(model_path + "/cnnLSTM_6units_structure.json", "w") as jason_file:
        jason_file.write(model_jason)

    model.save_weights(model_path + "/cnnLSTM_6units_weights.h5")
    print("Saved model to disk!")


def conv_lstm_predict(X_test):
    json_file = open('../saved_models/cnnLSTM_200_structure.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../saved_models/cnnLSTM_200_weights.h5")
    print(loaded_model.summary())
    print("Loaded model from disk!")

    y_pred = loaded_model.predict(X_test).flatten()
    print(y_pred)
    return y_pred

def test_on_sgRNA22_dataset():
    file_path = "../encoded_data/22sgRNA_validation2_data_for_testing.pkl"
    if os.path.exists(file_path):
        X, y = pkl.load(open(file_path, "rb"))
    else:
        X, y = encode_data.encode_22sgRNA_data()
        pkl.dump([X, y], open(file_path, "wb"))

    X_test = X.reshape(len(X), 1, 23, 4)
    y_pred = conv_lstm_predict(X_test)
    # pkl.dump(y_pred, open("./results/cnn_lstm_22sgRNA_results.pkl", "wb"))
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)

    plt.figure()
    # plt.subplot(221)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


test_on_sgRNA22_dataset()

