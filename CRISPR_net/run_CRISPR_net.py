# -*- coding: utf-8 -*-
# @Time     :1/5/19 10:10 PM
# @Auther   :Jason Lin
# @File     :run_CRISPR_net.py
# @Software :PyCharm
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas as pd
import numpy as np
import Encoder_sgRNA_off
import encode_data
pd.options.display.max_columns = None
from keras.models import model_from_json

def encode_on_off_seq_pairs(input_file = "./input_examples/on_off_seq_pairs_indels.txt"):
    inputs = pd.read_csv(input_file, delimiter=",", header=None, names=['on_seq', 'off_seq'])
    input_codes = []
    for idx, row in inputs.iterrows():
        on_seq = row['on_seq']
        off_seq = row['off_seq']
        while len(on_seq) < 24:
            on_seq = "-" + on_seq
        while len(off_seq) < 24:
            off_seq = "-" + off_seq
        en = Encoder_sgRNA_off.Encoder(sgRNA_seq=on_seq, off_seq=off_seq,
                                       with_category=True, label=-1, with_indel=True, dim_half=False)
        input_codes.append(en.on_off_code)
    input_codes = np.array(input_codes)
    input_codes = input_codes.reshape(len(input_codes), 1, 24, 7)
    y_pred = CRISPR_net_indels_predict(input_codes)
    inputs['CRISPR_net_score'] = y_pred
    print("\n-------------------------------- Result --------------------------------")
    print(inputs)
    print("------------------------------------------------------------------------")
    # inputs.to_csv("./CRISPR_net_results.csv", index=False)
    # print("Save the results to ./CRISPR_net_results.csv!")

def CRISPR_net_indels_predict(X_test):
    json_file = open("../saved_models/ConvLSTM_indel_structure_dim7_nnn.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../saved_models/ConvLSTM_indel_weights_full_data.h5")
    print("Loaded model from disk!")
    y_pred = loaded_model.predict(X_test).flatten()
    # print(y_pred)
    return y_pred

import argparse
# description="calculate X to the power of Y"
parser = argparse.ArgumentParser(description="CRISPR-Net v1.0 (Jan 10 2019)")
parser.add_argument("input_file", help="Example input file:\n GAGT_CCGAGCAGAAGAAGAATGG,GAGTACCAAGTAGAAGAAAAATTT\n"
                                       "GTTGCCCCACAGGGCAGTAAAGG,GTGGACACCCCGGGCAGGAAAGG")
args = parser.parse_args()
# print(args.square)
file = args.input_file
if not os.path.exists(args.input_file):
    print("File doesn't exist!")
    os.system("python run_CRISPR_net.py -h")
else:
    encode_on_off_seq_pairs(file)
