# -*- coding: utf-8 -*-
# @Time     :1/3/19 4:26 PM
# @Auther   :Jason Lin
# @File     :CIRCLE_seq_mix_logocv.py
# @Software :PyCharm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle as pkl
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

targetsite_rna = pd.read_csv("../data/CIRCLE_seq_gRNA_dict.csv")
tsite_dict = {}
for idx, row in targetsite_rna.iterrows():
    tsite_dict[row['gRNA_seq']] = row['Targetsite']
print(tsite_dict)


def conv_lstm_predict(X_test, weight_file,structure_file):
    from keras.models import model_from_json
    json_file = open(structure_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_file)
    print("Loaded model from disk!")
    y_pred = loaded_model.predict(X_test).flatten()
    print(y_pred)
    return y_pred

def circle_seq_logv_roc():
    import glob
    import re
    import os
    import encode_data
    from scipy import interp
    from sklearn import metrics
    file_path = "../encoded_data/CIRCLE_seq_data_Mix.pkl"
    if os.path.exists(file_path):
        X, y, read, sgRNA_types = pkl.load(open(file_path, "rb"))
    else:
        X, y, read, sgRNA_types = encode_data.encode_CIRCLE_data(type='Mix')
        X = X.reshape((len(X), 1, 24, 7))
        pkl.dump([X, y, read, sgRNA_types], open(file_path, "wb"))

    path = "../saved_models/"
    weights_files = glob.glob(path + "ConvLSTM_indel_weights_dim7_nnn*.h5")
    structure_files = glob.glob(path + "ConvLSTM_indel_structure_dim7_nnn.json")
    print(weights_files)
    print(structure_files)
    structure = structure_files[0]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(weights_files)):
        weights = weights_files[i]
        rna_seq = re.split("_|\.", weights)[-2]
        print(rna_seq)
        train_idx = sgRNA_types != rna_seq
        test_idx = sgRNA_types == rna_seq
        X_train = X[train_idx]
        # y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        print("training data:", len(X_train), " testing data:", len(X_test))

        y_pred = conv_lstm_predict(X_test, weight_file=weights, structure_file=structure)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        # fpr = np.round(fpr, 3)
        # tpr = np.round(tpr, 3)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        print(rna_seq, roc_auc)
        # f = open("./cnn_lstm_result_80dense.txt", "a+")
        # print(rna_seq, roc_auc, file=f)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.6,
                 label='%s (AUC = %0.4f)' % (tsite_dict[rna_seq], roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    print(aucs)
    print(mean_auc)

    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f)' % (mean_auc),
             lw=2, alpha=.8)
    # pkl.dump([mean_fpr, mean_tpr, mean_auc, std_auc], open("./results/circle_cnn_lstm_roc.pkl", "wb"))
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(linestyle=':', lw=1.5)
    plt.tick_params(axis='x', labelcolor="black")
    plt.tick_params(axis='y', labelcolor="black")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.tight_layout()
    # plt.savefig("./circle_logocv_roc.png")
    plt.show()

circle_seq_logv_roc()