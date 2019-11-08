# -*- coding: utf-8 -*-
# @Time     :9/19/19 9:30 PM
# @Auther   :Jason Lin
# @File     :CRISPR_Net_Aggregate.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import Encoder_sgRNA_off
from keras.models import model_from_json
from sklearn.externals import joblib
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def encode_on_off_seq_pairs(on_seqs, off_seqs):
    input_codes = []
    for i in range(len(on_seqs)):
        on_seq = on_seqs[i]
        off_seq = off_seqs[i]
        while len(on_seq) < 24:
            on_seq = "-" + on_seq
        while len(off_seq) < 24:
            off_seq = "-" + off_seq
        en = Encoder_sgRNA_off.Encoder(sgRNA_seq=on_seq, off_seq=off_seq,
                                       with_category=True, label=-1, with_indel=True, dim_half=False)
        input_codes.append(en.on_off_code)
    input_codes = np.array(input_codes)
    input_codes = input_codes.reshape((len(input_codes), 1, 24, 7))
    # y_pred = CRISPR_net_indels_predict(input_codes)
    return input_codes

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

def build_aggregate_feature(pred_score, isgenic):
    # isgenic = ~df['Gene_mark'].isnull()
    type = ""
    crispr_val = pred_score
    feature_dict = dict()
    crispr_val_genic = crispr_val[isgenic == 1]
    crispr_val_non_genic = crispr_val[isgenic == 0]
    total = len(pred_score)
    feature_list = []
    # isgenic feature
    feature_dict['sum_genic' + type] = sum(crispr_val_genic)
    feature_dict['mean_genic' + type] = np.mean(crispr_val_genic)
    feature_dict['num_genic' + type] = len(crispr_val_genic)
    feature_dict['sd_genic' + type] = np.std(crispr_val_genic)
    feature_dict['var_genic' + type] = np.var(crispr_val_genic)
    feature_dict['per99_genic' + type] = np.percentile(crispr_val_genic, 99)
    feature_dict['per95_genic' + type] = np.percentile(crispr_val_genic, 95)
    feature_dict['per90_no_genic' + type] = np.percentile(crispr_val_genic, 90)
    # non-genic feature
    feature_dict['sum_no_genic' + type] = sum(crispr_val_non_genic)
    feature_dict['mean_no_genic' + type] = np.mean(crispr_val_non_genic)
    feature_dict['num_no_genic' + type] = len(crispr_val_non_genic)
    feature_dict['sd_no_genic' + type] = np.std(crispr_val_non_genic)
    feature_dict['var_no_genic' + type] = np.var(crispr_val_non_genic)
    feature_dict['per99_no_genic' + type] = np.percentile(crispr_val_non_genic, 99)
    feature_dict['per95_no_genic' + type] = np.percentile(crispr_val_non_genic, 95)
    feature_dict['per90_no_genic' + type] = np.percentile(crispr_val_non_genic, 90)
    # genic/no_genic feature
    feature_dict['frac_genic' + type] = feature_dict['sum_genic' + type] / total
    feature_dict['frac_non_genic' + type] = feature_dict['sum_no_genic' + type] / total
    feature_dict['div_genic' + type] = feature_dict['sum_genic' + type] / feature_dict['sum_no_genic' + type]
    feature_dict['div_mean_genic' + type] = feature_dict['mean_genic' + type] / feature_dict['mean_no_genic' + type]
    # whole dataset feature
    feature_dict['sum' + type] = sum(crispr_val)
    feature_dict['mean' + type] = np.mean(crispr_val)
    feature_dict['var' + type] = np.var(crispr_val)
    feature_dict['sd' + type] = np.std(crispr_val)
    feature_df = pd.DataFrame(feature_dict, index=[0])
    # print(feature_df.values)
    # print(feature_df)
    return feature_df.values

def run_CRISPR_net_aggregate(on_seqs, off_seqs, isgenic):
    encoded_seqs = encode_on_off_seq_pairs(on_seqs, off_seqs)
    pred_score = CRISPR_net_indels_predict(encoded_seqs)
    print(pred_score.shape)
    features = build_aggregate_feature(pred_score, isgenic)
    aggregate = joblib.load('../saved_models/CRISPR-Net-Aggregate.m')
    aggregate_score = aggregate.predict(features)
    # print(aggregate_score)
    return aggregate_score

def main(file):
    # offtarget_df = pd.read_csv("./input_examples/aggregate_example_GACCTTGCATTGTACCCGAG.csv")
    offtarget_df = pd.read_csv(file)
    gRNA_seq = offtarget_df['on_target']
    offtarget_seq = offtarget_df['off_target']
    genic = offtarget_df['Gene_mark']
    # print(genic)
    genic_labels = np.ones(len(genic))
    genic_labels[genic.isnull()] = 0
    # print(genic_labels)
    aggregate_score = run_CRISPR_net_aggregate(gRNA_seq, offtarget_seq, genic_labels)
    print(aggregate_score)

import argparse
parser = argparse.ArgumentParser(description="CRISPR-Net-Aggregate v1.0 (Aug 10 2019)")
parser.add_argument("input_file", help="input_file.csv must have three columns (on_target, off_target and Gene_mark)")
args = parser.parse_args()
file = args.input_file
if not os.path.exists(args.input_file):
    print("File doesn't exist!")
    os.system("python CRISPR_Net.py -h")
# elif(len(args) < 3):
#     os.system("python CRISPR_Net.py -h")
else:
    main(file)




