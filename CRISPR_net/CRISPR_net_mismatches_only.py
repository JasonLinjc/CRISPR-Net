# -*- coding: utf-8 -*-
# @Time     :1/5/19 10:11 PM
# @Auther   :Jason Lin
# @File     :CRISPR_net_mismatches_only.py
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
        if len(on_seq) != 23 or len(off_seq) != 23:
            print("The length of the sequence pair should be 23!")
            return 0
        en = Encoder_sgRNA_off.Encoder(sgRNA_seq=on_seq, off_seq=off_seq,
                                       with_category=True, label=-1, with_indel=False, dim_half=True)
        input_codes.append(en.on_off_code)
    input_codes = np.array(input_codes)
    input_codes = input_codes.reshape(len(input_codes), 1, 23, 4)
    y_pred = CRISPR_net_indels_predict(input_codes)
    inputs['CRISPR_net_score'] = y_pred
    print("\n-------------------------------- Result --------------------------------")
    print(inputs)
    print("------------------------------------------------------------------------")
    inputs.to_csv("./CRISPR_net_results_mismatches_only.csv", index=False)
    print("Save the results to ./CRISPR_net_results.csv!")

def CRISPR_net_indels_predict(X_test):
    json_file = open("../saved_models/cnnLSTM_200_structure.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../saved_models/cnnLSTM_200_weights.h5")
    print("Loaded model from disk!")
    y_pred = loaded_model.predict(X_test).flatten()
    # print(y_pred)
    return y_pred

import argparse
# description="calculate X to the power of Y"
parser = argparse.ArgumentParser(description="CRISPR-Net v1.0 (Jan 10 2019)")
parser.add_argument("input_file", help="input_file example:\n"                  
                                       "GATGGTAGATGGAGACTCAGAGG,GGTAGGAAATGGAGAGGCAGAGG\n"
                                       "GGGTGGGGGGAGTTTGCTCCCGG,GTGTGGGGTAAATTTGCTCCCAG")
args = parser.parse_args()
# print(args.square)
file = args.input_file
if not os.path.exists(args.input_file):
    print("File doesn't exist!")
    os.system("python CRISPR_net_mismatches_only.py -h")
else:
    encode_on_off_seq_pairs(file)