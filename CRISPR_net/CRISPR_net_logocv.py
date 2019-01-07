# -*- coding: utf-8 -*-
# @Time     :1/5/19 9:37 PM
# @Auther   :Jason Lin
# @File     :CRISPR_net_logocv.py
# @Software :PyCharm

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

import keras
import numpy as np
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, LSTM, BatchNormalization, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Model, model_from_json
from sklearn.model_selection import StratifiedKFold
import encode_data
import os
from scipy import interp
import pandas as pd
import pickle as pkl
from sklearn import metrics
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt
from numpy.random import seed
import time
seed(6)
from tensorflow import set_random_seed
set_random_seed(66)


targetsite_rna = pd.read_csv("../data/CIRCLE_seq_gRNA_dict.csv")
tsite_dict = {}
for idx, row in targetsite_rna.iterrows():
    tsite_dict[row['gRNA_seq']] = row['Targetsite']
print(tsite_dict)
if os.path.isdir('../encoded_data'):
    pass
else:
    os.mkdir('../encoded_data')


def ConvLSTM_indel(X_train, y_train, model_name="logv", using_pre_train=False):

    inputs = Input(shape=(1, 24, 7), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding="same", activation='relu')(inputs)
    conv_2 = Conv2D(10, (1, 2), padding="same", activation='relu')(inputs)
    conv_3 = Conv2D(10, (1, 3), padding="same", activation='relu')(inputs)
    conv_4 = Conv2D(10, (1, 5), padding="same", activation='relu')(inputs)

    bn_1 = BatchNormalization()(conv_1)
    bn_2 = BatchNormalization()(conv_2)
    bn_3 = BatchNormalization()(conv_3)
    bn_4 = BatchNormalization()(conv_4)

    bn_1 = Reshape((24, 10))(bn_1)
    bn_2 = Reshape((24, 10))(bn_2)
    bn_3 = Reshape((24, 10))(bn_3)
    bn_4 = Reshape((24, 10))(bn_4)

    # BiLSTM
    blstm_1 = Bidirectional(LSTM(6, return_sequences=True, input_shape=(23, 10)))(bn_1)
    blstm_2 = Bidirectional(LSTM(6, return_sequences=True, input_shape=(23, 10)))(bn_2)
    blstm_3 = Bidirectional(LSTM(6, return_sequences=True, input_shape=(23, 10)))(bn_3)
    blstm_4 = Bidirectional(LSTM(6, return_sequences=True, input_shape=(23, 10)))(bn_4)

    bn_output =  keras.layers.concatenate([blstm_1, blstm_2, blstm_3, blstm_4])

    flatten_output = Flatten()(bn_output)

    x = Dense(80, activation='relu')(flatten_output)
    x = Dense(20, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.45)(x)
    prediction = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs, prediction)
    adam_opt = keras.optimizers.Adam(lr=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=adam_opt)
    print(model.summary())

    model.fit(X_train, y_train, batch_size=10000, epochs=300, shuffle=True)
    # print(time.clock() - start_time)

    model_path = "./logocv_models"
    if os.path.isdir(model_path):
        pass
    else:
        os.mkdir(model_path)

    model_jason = model.to_json()
    with open(model_path + "/CRISPR_net_indel_structure_"+ model_name + ".json", "w") as jason_file:
        jason_file.write(model_jason)

    model.save_weights(model_path + "/CRISPR_net_indel_weights_"+ model_name + ".h5")
    print("Saved model to disk!")

def conv_lstm_predict(X_test, model_name="logv"):
    json_file = open("./logocv_models/CRISPR_net_indel_structure_" +  model_name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./logocv_models/CRISPR_net_indel_weights_"+ model_name + ".h5")
    print("Loaded model from disk!")

    y_pred = loaded_model.predict(X_test).flatten()
    print(y_pred)
    return y_pred

def logocv_on_circle_data():
    file_path = "../encoded_data/CIRCLE_seq_data_Mix.pkl"
    if os.path.exists(file_path):
        X, y, read, sgRNA_types = pkl.load(open(file_path, "rb"))
    else:
        X, y, read, sgRNA_types = encode_data.encode_CIRCLE_data(type='Mix')
        X = X.reshape((len(X), 1, 24, 7))
        pkl.dump([X, y, read, sgRNA_types], open(file_path, "wb"))

    sgrna = np.array(list(set(sgRNA_types)))
    sgrna = sgrna[::-1]
    print(len(sgrna))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    seq_list = []
    print(len(sgrna))
    for seq in sgrna:
        print(seq)
        train_idx = sgRNA_types != seq
        test_idx = sgRNA_types == seq
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        seq_list.append(seq)

        print("training data:", len(X_train), " testing data:", len(X_test))

        ConvLSTM_indel(X_train, y_train, model_name="dim7_mix_" + seq)
        y_pred = conv_lstm_predict(X_test, model_name="dim7_mix_" + seq)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        print(seq, roc_auc)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='%s (AUC = %0.4f)' % (tsite_dict[seq], roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    print(aucs)
    print(mean_auc)

    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.savefig("./logv_on_circle_excluded_nnn1.png")
    plt.show()

logocv_on_circle_data()