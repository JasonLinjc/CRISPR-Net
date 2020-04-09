# -*- coding: utf-8 -*-
# @Time     :10/17/18 4:04 PM
# @Auther   :Jason Lin
# @File     :encode_CD33$.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import Encoder_sgRNA_off
import pickle as pkl
pd.options.display.max_columns = None

def load_elevation_CD33_dataset():
    print("Loading dataset II/1...")
    cd33_data = pd.read_pickle("../data/Dataset II (mismatch-only)"
                               + "/Listgarten_ElevationDataset (dataset II-1&II-2&II-4)/cd33 (dataset II-1).pkl")
    cd33_mut = cd33_data[0]
    cd33_code = []
    label = []
    for idx, row in cd33_mut.iterrows():
        on_seq = row['30mer']
        off_seq = row['30mer_mut']
        etp_val = row['Day21-ETP']
        etp_label = row['Day21-ETP-binarized']
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_reg_val=True, value=etp_val)
        cd33_code.append(en.on_off_code)
        label.append(etp_label)
    label = np.array(label)
    cd33_code = np.array(cd33_code)
    print("Finished!", cd33_code.shape, len(label[label>0]))
    return cd33_code, np.array(label)

def load_elevation_hmg_dataset():
    print("Loading dataset II/2...")
    hmg_data = pd.read_pickle("../data/Dataset II (mismatch-only)/Listgarten_ElevationDataset (dataset II-1&II-2&II-4)/hmg_data (dataset II-2).pkl")
    hmg_code = []
    hmg_vals = []
    for idx, row in hmg_data.iterrows():
        on_seq = row['30mer']
        off_seq = row['30mer_mut']
        reg_val = row['readFraction']
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_reg_val=True, value=reg_val)
        hmg_code.append(en.on_off_code)
        hmg_vals.append(en.value)
        # print(en.sgRNA_seq)
        # print(en.off_seq)
        # print(en.sgRNA_off_code)

    hmg_vals = np.array(hmg_vals)
    hmg_code = np.array(hmg_code)
    # print(len(hmg_vals[hmg_vals>0]))
    hmg_label = np.zeros(len(hmg_vals))
    hmg_label[hmg_vals>0] = 1
    print("Finished!", "dataset size: ", hmg_code.shape, len(hmg_label[hmg_label>0]))
    return np.array(hmg_code), np.array(hmg_label)

def load_elevation_guideseq_data():
    print("Loading dataset II/4...")
    guideseq_data = pd.read_pickle("../data/Dataset II (mismatch-only)/Listgarten_ElevationDataset (dataset II-1&II-2&II-4)/guideseq_data (dataset II-3).pkl")
    guideseq_code = []
    guideseq_vals = []
    for idx, row in guideseq_data.iterrows():
        on_seq = row['30mer']
        off_seq = row['30mer_mut']
        reg_val = row['GUIDE-SEQ Reads']
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_reg_val=True, value=reg_val)
        guideseq_code.append(en.on_off_code)
        guideseq_vals.append(en.value)

    guideseq_code = np.array(guideseq_code)
    guideseq_vals = np.array(guideseq_vals)
    guideseq_labels = np.zeros(len(guideseq_vals))
    guideseq_labels[guideseq_vals > 0] = 1
    print("Dataset size:", guideseq_code.shape, "positive num:", len(guideseq_labels[guideseq_labels > 0]))
    return np.array(guideseq_code), np.array(guideseq_labels)

def load_22sgRNA_data():
    print("Loading Listgarten dataset II/6...")
    sgRNA22_data = pd.read_csv("../data/Dataset II (mismatch-only)/dataset II-6/Listgarten_22gRNA_wholeDataset.csv")
    sgRNA22_code = []
    sgRNA22_labels = []
    for idx, row in sgRNA22_data.iterrows():
        on_seq = row['sgRNA_seq'].upper()
        # print(idx, on_seq)
        off_seq = row['off_seq'].upper()
        label = row['label']
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        sgRNA22_code.append(en.on_off_code)
        sgRNA22_labels.append(en.label)
    sgRNA22_labels = np.array(sgRNA22_labels)
    sgRNA22_code = np.array(sgRNA22_code)
    print("Finished!", "Dataset size: ", np.array(sgRNA22_code).shape, len(sgRNA22_labels[sgRNA22_labels > 0]))
    return np.array(sgRNA22_code), np.array(sgRNA22_labels)

def load_CIRCLE_data():
    print("Encoding CIRCLE-seq dataset (dataset II/1)...")
    circle_data = pd.read_csv("../data/Dataset I (indel&mismatch)/dataset I-1/CIRCLE_seq_10gRNA_wholeDataset.csv")
    circle_codes = []
    circle_labels = []
    for idx, row in circle_data.iterrows():
        on_seq = row['sgRNA_seq']
        off_seq = row['off_seq']
        label = row['label']
        read_val = row['Read']
        # mut_type = row['mut_type']
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        circle_codes.append(en.on_off_code)
        circle_labels.append(label)
        # mut_types.append(mut_type)
        # infos.append([sgRNA_seq, off_seq, read_val, mut_type, label, sgRNA_type])
        # print(idx, mut_type, label, sgRNA_seq, off_seq, read_val, sgRNA_type)
    circle_codes = np.array(circle_codes)
    circle_labels = np.array(circle_labels)
    print("Finished!", "Dataset size:", circle_codes.shape, len(circle_labels[circle_labels>0]))
    return circle_codes, circle_labels

def load_Kleinstiver_data():
    print("Loading Kleinsitver dataset (dataset II/5)...")
    sgRNA5_data = pd.read_csv("../data/Dataset II (mismatch-only)/dataset II-5/Kleinstiver_5gRNA_wholeDataset.csv")
    sgRNA5_code = []
    sgRNA5_labels = []
    for idx, row in sgRNA5_data.iterrows():
        on_seq = row['sgRNA_seq'].upper()
        off_seq = row['off_seq'].upper()
        #  print(idx, on_seq)
        label = row['label']
        en = Encoder_sgRNA_off.Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        sgRNA5_code.append(en.on_off_code)
        sgRNA5_labels.append(en.label)
    sgRNA5_labels = np.array(sgRNA5_labels)
    sgRNA5_code = np.array(sgRNA5_code)
    print("Finished!")
    print(sgRNA5_code.shape, len(sgRNA5_labels[sgRNA5_labels > 0]))
    return sgRNA5_code, sgRNA5_labels

def encode_seq(seq):
    encoded_dict = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1],'-': [0, 0, 0, 0, 0]}
    code_list = []
    for i in range(len(seq)):
        code_list.append(encoded_dict[seq[i]])
    return np.array(code_list)

def load_siteseq_data():
    print("Loading SITE-Seq dataset (dataset II/3) .....")
    siteseq_data = pd.read_csv("../data/Dataset II (mismatch-only)/dataset II-3/SITE-Seq_offTarget_wholeDataset.csv", index_col=0)
    ###############
    code = []
    gRNA = []
    reads = []
    ###############
    direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, 'N': 1, '-': 0}
    for idx, row in siteseq_data.iterrows():
        on_seq = '-'+row['on_seq'].upper()
        off_seq = '-'+row['off_seq'].upper()
        on_off_dim6_codes = []
        on_bases = list(on_seq)
        off_bases = list(off_seq)
        on_bases[-3] = off_bases[-3]
        on_codes = encode_seq(on_seq)
        off_codes = encode_seq(off_seq)
        for i in range(len(on_bases)):
            # print(len(on_bases))
            on_b = on_bases[i]
            off_b = off_bases[i]
            diff_code = np.bitwise_or(on_codes[i], off_codes[i])
            dir_code = np.zeros(2)
            if direction_dict[on_b] == direction_dict[off_b]:
                pass
            elif direction_dict[on_b] > direction_dict[off_b]:
                dir_code[0] = 1.0
            else:
                dir_code[1] = 1.0
            on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))

        code.append(on_off_dim6_codes)
        gRNA.append(row['on_seq'])
        reads.append(row['reads'])
    code = np.array(code)
    reads = np.array(reads)
    labels = np.zeros(len(reads))
    labels[reads > 0] = 1
    print(code.shape, len(labels[labels>0]))
    # pkl.dump([code, gRNA, reads], open("./siteseq_9gRNA_code_dim7_dedup.pkl", "wb"))
    return code, labels

# encode_siteseq_data()
# load_siteseq_data()

