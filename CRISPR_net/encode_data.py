# -*- coding: utf-8 -*-
# @Time     :10/17/18 4:04 PM
# @Auther   :Jason Lin
# @File     :encode_CD33$.py
# @Software :PyCharm

import pandas as pd
import numpy as np
import Encoder_sgRNA_off
pd.options.display.max_columns = None

def encode_CD33_mut(with_indel=False, dim_half=True):
    print("Loading CD33 dataset...")
    cd33_data = pd.read_pickle("../data/cd33.pkl")
    cd33_mut = cd33_data[0]
    cd33_code = []
    label = []
    for idx, row in cd33_mut.iterrows():
        sgRNA_seq = row['30mer']
        off_seq = row['30mer_mut']
        if with_indel:
            sgRNA_seq = "_" + sgRNA_seq
            off_seq = "_" + off_seq
        etp_val = row['Day21-ETP']
        etp_label = row['Day21-ETP-binarized']
        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=etp_val, with_indel=with_indel)
            cd33_code.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=etp_val,
                                           with_indel=with_indel, dim_half=False)
            cd33_code.append(en.on_off_code)
        label.append(en.value)
    print("Finished!")
    return np.array(cd33_code), np.array(label)

def encode_crispor():
    crispor_data = pd.read_csv("../data/crispor_allscore.csv")
    crispor_codes = []
    labels = []
    for idx, row in crispor_data.iterrows():
        sgRNA_seq = row['wt_seq']
        off_seq = row['off_seq']
        label = row['label']
        en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label)
        crispor_codes.append(en.sgRNA_off_code)
        labels.append(en.label)
    return np.array(crispor_codes), np.array(labels)

def encode_penghui_data():
    penghui_data = pd.read_csv("../data/penghui_dataset.csv")
    penghui_code = []
    labels = []
    for idx, row in penghui_data.iterrows():
        # print("-------", idx, "-------")
        sgRNA_seq = row['sgRNA']
        off_seq = row['offtarget']
        label = row['label']
        en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label)
        penghui_code.append(en.sgRNA_off_code)
        labels.append(label)
    print("Penghui dataset is loaded!")
    return np.array(penghui_code), np.array(labels)


def build_22sgRNA_data():
    t_offs = pd.read_excel("../data/22sgRNA_data.xlsx", sheet_name="GUIDE-seq output")
    t_offs = t_offs[t_offs['Targetsite'] != "EMX1_site1"]
    t_offs = t_offs[t_offs['Mismatches'] > 0]

    p_offs = pd.read_csv("../data/validation2_6misNN_casoffinder.txt", delimiter="\t")
    # p_offs = p_offs[p_offs['Mismatches'] <= 7]

    p_off_list = []
    t_off_list = []
    for idx, row in p_offs.iterrows():
        str = row['crRNA'] + row['DNA']
        str = str.upper()
        p_off_list.append(str)

    for idx, row in t_offs.iterrows():
        str = row['Target Sequence'] + row['Off-target Sequence']
        str = str.upper()
        t_off_list.append(str)


    print(t_off_list[0])
    print(p_off_list[0])

    # Find overlapping sgRNA-Off pairs
    overlapping_flags = []

    for p_off in p_off_list:
        if p_off in t_off_list:
            overlapping_flags.append(1)
        else:
            overlapping_flags.append(0)

    overlapping_flags = np.array(overlapping_flags)
    print(len(overlapping_flags[overlapping_flags == 1]))
    print(len(t_off_list))

    p_offs['overlapping_flag'] = np.array(overlapping_flags)

    p_offs = p_offs[p_offs['overlapping_flag'] == 0]

    fake_df = pd.DataFrame()
    true_df = pd.DataFrame()

    fake_df['sgRNA_seq'] = p_offs['crRNA']
    fake_df['off_seq'] = p_offs['DNA']
    fake_df['label'] = np.zeros((len(p_offs)))
    fake_df['read'] = np.zeros((len(p_offs)))

    true_df['sgRNA_seq'] = t_offs['Target Sequence']
    true_df['off_seq'] = t_offs['Off-target Sequence']
    true_df['label'] = np.ones((len(t_offs)))
    true_df['read'] = t_offs['GUIDE-seq read counts']
    data = pd.concat([fake_df, true_df])
    data.to_csv("../data/22sgRNA_off_data_6mis_new.csv", index=False)

def encode_22sgRNA_data(with_indel = False, dim_half=True):
    print("Loading sgRNA22 dataset...")
    sgRNA22_data = pd.read_csv("../data/22sgRNA_off_data_6mis_new.csv")
    sgRNA22_code = []
    sgRNA22_labels = []
    for idx, row in sgRNA22_data.iterrows():
        sgRNA_seq = row['sgRNA_seq'].upper()
        off_seq = row['off_seq'].upper()
        if with_indel:
            sgRNA_seq = "_" + sgRNA_seq
            off_seq = "_" + off_seq
        label = row['label']
        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label,
                                       with_indel=with_indel)
            sgRNA22_code.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label,
                                           with_indel=with_indel, dim_half=False)
            sgRNA22_code.append(en.on_off_code)
        sgRNA22_labels.append(en.label)
    print("Finished!")
    print(np.array(sgRNA22_code).shape)
    return np.array(sgRNA22_code), np.array(sgRNA22_labels)

def encode_ele_guideseq_data(with_indel=False, dim_half=True):
    print("Loading elevation guideseq dataset...")
    guideseq_data = pd.read_pickle("../data/guideseq_data.pkl")
    guideseq_code = []
    guideseq_vals = []
    for idx, row in guideseq_data.iterrows():
        sgRNA_seq = row['30mer']
        off_seq = row['30mer_mut']
        if with_indel:
            sgRNA_seq = "_" + sgRNA_seq
            off_seq = "_" + off_seq
        reg_val = row['GUIDE-SEQ Reads']
        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=reg_val, with_indel=with_indel)
            guideseq_code.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=reg_val,
                                           with_indel=with_indel, dim_half=False)
            guideseq_code.append(en.on_off_code)
        guideseq_vals.append(en.value)
        # print(en.sgRNA_seq)
        # print(en.off_seq)
        # print(en.sgRNA_off_code)
    print("Finished!")
    return np.array(guideseq_code), np.array(guideseq_vals)

def encode_ele_hmg_data(with_indel=False, dim_half=True):
    print("Loading elevation hmg dataset...")
    guideseq_data = pd.read_pickle("../data/hmg_data.pkl")
    guideseq_code = []
    guideseq_vals = []
    for idx, row in guideseq_data.iterrows():
        sgRNA_seq = row['30mer']
        off_seq = row['30mer_mut']
        if with_indel:
            sgRNA_seq = "_" + sgRNA_seq
            off_seq = "_" + off_seq
        reg_val = row['readFraction']
        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=reg_val, with_indel=with_indel)
            guideseq_code.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=reg_val,
                                           with_indel=with_indel, dim_half=False)
            guideseq_code.append(en.on_off_code)

        guideseq_vals.append(en.value)
        # print(en.sgRNA_seq)
        # print(en.off_seq)
        # print(en.sgRNA_off_code)
    print("Finished!")
    return np.array(guideseq_code), np.array(guideseq_vals)


def encode_cd33_indels_data(dim_half=True):
    print("Loading CD33 with indels dataset...")
    cd33_data = pd.read_csv("../data/cd33_with_indel_seq_data.csv")
    # print(cd33_data)
    cd33_codes = []
    cd33_reg_vals = []
    for idx, row in cd33_data.iterrows():
        sgRNA_seq = row['sgRNA_seq']
        off_seq = row['off_seq']
        reg_val = row['etp_normalised']

        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=reg_val, with_indel=True)
            cd33_codes.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_reg_val=True, value=reg_val,
                                           with_indel=True, dim_half=False)
            cd33_codes.append(en.on_off_code)
        cd33_reg_vals.append(en.value)
    cd33_codes = np.array(cd33_codes)
    cd33_reg_vals = np.array(cd33_reg_vals)
    print("Finished!")
    return cd33_codes, cd33_reg_vals


def encode_CIRCLE_data(dim_half=True):

    print("Encoding CIRCLE-seq dataset...")
    circle_data = pd.read_csv("../data/CIRCLE_seq_whole_data_excluded_nnn_info.csv")
    circle_codes = []
    circle_labels = []
    reads = []
    sgRNA_types = []
    mut_types = []
    seq_pairs = []
    infos = []
    for idx, row in circle_data.iterrows():
        sgRNA_seq = row['sgRNA_seq']
        off_seq = row['off_seq']
        label = row['label']
        read_val = row['Read']
        sgRNA_type = row['sgRNA_type']
        # mut_type = row['mut_type']

        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq = sgRNA_seq, off_seq=off_seq, with_category=True, label=label, with_indel=True)
            circle_codes.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label, with_indel=True, dim_half=False)
            circle_codes.append(en.on_off_code)
        circle_labels.append(label)
        seq_pairs.append([sgRNA_seq, off_seq])
        reads.append(read_val)
        sgRNA_types.append(sgRNA_type)
        # mut_types.append(mut_type)
        # infos.append([sgRNA_seq, off_seq, read_val, mut_type, label, sgRNA_type])
        # print(idx, mut_type, label, sgRNA_seq, off_seq, read_val, sgRNA_type)
    print("Finished!")
    reads = np.array(reads)
    circle_codes = np.array(circle_codes)
    circle_labels = np.array(circle_labels)
    sgRNA_types = np.array(sgRNA_types)
    mut_types = np.array(mut_types)
    seq_pairs = np.array(seq_pairs)
    infos = np.array(infos)
    return circle_codes, circle_labels, reads, sgRNA_types


def encode_case_study_indel_data(dim_half=True):

    print("Encoding CIRCLE-seq dataset...")
    circle_data = pd.read_csv("/home/jieconlin3/year2semA/CRISPR_off_target/CRISPR-Net/case_study/cs_CasOffinder.csv")
    circle_codes = []
    circle_labels = []
    reads = []
    sgRNA_types = []
    mut_types = []
    seq_pairs = []
    infos = []
    for idx, row in circle_data.iterrows():
        sgRNA_seq = row['sgRNA_seq']
        off_seq = row['off_seq']
        label = row['label']
        read_val = row['Read']
        sgRNA_type = row['sgRNA_type']
        mut_type = row['mut_type']

        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq = sgRNA_seq, off_seq=off_seq, with_category=True, label=label, with_indel=True)
            circle_codes.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label, with_indel=True, dim_half=False)
            circle_codes.append(en.on_off_code)
        circle_labels.append(label)
        seq_pairs.append([sgRNA_seq, off_seq])
        reads.append(read_val)
        sgRNA_types.append(sgRNA_type)
        mut_types.append(mut_type)
        infos.append([sgRNA_seq, off_seq, read_val, mut_type, label, sgRNA_type])
        print(idx, mut_type, label, sgRNA_seq, off_seq, read_val, sgRNA_type)
    print("Finished!")
    reads = np.array(reads)
    circle_codes = np.array(circle_codes)
    circle_labels = np.array(circle_labels)
    sgRNA_types = np.array(sgRNA_types)
    mut_types = np.array(mut_types)
    seq_pairs = np.array(seq_pairs)
    infos = np.array(infos)
    return circle_codes, circle_labels, reads, sgRNA_types, infos

def encode_Haeussler_mm6():
    print("Loading Haeussler dataset...")
    h_data = pd.read_csv("../data/Haeussler_mm6_scores.csv", index_col='idx')
    # print(h_data)
    code_list = []
    label_list = []
    for idx, row in h_data.iterrows():
        print("---------", idx, "----------")
        sgRNA_seq = row['on']
        off_seq = row['off']
        if row['valid']:
            label = 1.0
        else:
            label = 0.0
        en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label, with_indel=False)
        code_list.append(en.sgRNA_off_code)
        label_list.append(label)
    code_list = np.array(code_list)
    label_list = np.array(label_list)
    print(code_list.shape)
    print(label_list.shape)
    return code_list, label_list

def build_validatation1_data():
    true_offs = pd.read_csv("../data/validatation1_5sgRNA.tsv", delimiter="\t")
    offinder_offs = pd.read_csv("../data/validation1_5sgRNA_6misNN_casoffinder.txt", delimiter="\t")

    p_offs = offinder_offs
    t_offs = true_offs
    print(t_offs)

    p_off_list = []
    t_off_list = []
    for idx, row in p_offs.iterrows():
        str = row['crRNA'] + row['DNA']
        str = str.upper()
        p_off_list.append(str)

    for idx, row in t_offs.iterrows():
        str = row['Target Sequence'] + row['Off-Target Sequence']
        str = str.upper()
        t_off_list.append(str)

    # Find overlapping sgRNA-Off pairs
    overlapping_flags = []
    for p_off in p_off_list:
        if p_off in t_off_list:
            overlapping_flags.append(1)
        else:
            overlapping_flags.append(0)

    overlapping_flags = np.array(overlapping_flags)
    print(len(overlapping_flags[overlapping_flags == 1]))
    print(len(t_off_list))

    p_offs['overlapping_flag'] = np.array(overlapping_flags)

    p_offs = p_offs[p_offs['overlapping_flag'] == 0]

    fake_df = pd.DataFrame()
    true_df = pd.DataFrame()

    fake_df['sgRNA_seq'] = p_offs['crRNA']
    fake_df['off_seq'] = p_offs['DNA']
    fake_df['label'] = np.zeros((len(p_offs)))

    true_df['sgRNA_seq'] = t_offs['Target Sequence']
    true_df['off_seq'] = t_offs['Off-Target Sequence']

    true_df['label'] = np.ones((len(t_offs)))
    data = pd.concat([fake_df, true_df])
    print(data)
    data.to_csv("../data/validation1_5sgRNA_off_data_6mis_new.csv", index=False)


def encode_validation1_data(with_indel = False, dim_half=True):
    print("Loading validation1 5 sgRNA dataset...")
    sgRNA22_data = pd.read_csv("../data/validation1_5sgRNA_off_data_6mis_new.csv")
    sgRNA22_code = []
    sgRNA22_labels = []
    for idx, row in sgRNA22_data.iterrows():
        sgRNA_seq = row['sgRNA_seq'].upper()
        off_seq = row['off_seq'].upper()
        if with_indel:
            sgRNA_seq = "_" + sgRNA_seq
            off_seq = "_" + off_seq
        label = row['label']
        if dim_half:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label,
                                       with_indel=with_indel)
            sgRNA22_code.append(en.sgRNA_off_code)
        else:
            en = Encoder_sgRNA_off.Encoder(sgRNA_seq=sgRNA_seq, off_seq=off_seq, with_category=True, label=label,
                                           with_indel=with_indel, dim_half=False)
            sgRNA22_code.append(en.on_off_code)
        sgRNA22_labels.append(en.label)
    print("Finished!")
    print(np.array(sgRNA22_code).shape)
    return np.array(sgRNA22_code), np.array(sgRNA22_labels)

# build_22sgRNA_data()
# encode_Haeussler_mm6()
# encode_validation1_data()
"""
codes, vals = encode_ele_hmg_data(with_indel=True)
print(codes.shape)
print(vals.shape)
codes, vals = encode_CD33_mut(with_indel=True)
print(codes.shape)
print(vals.shape)
codes, vals = encode_ele_guideseq_data(with_indel=True)
print(codes.shape)
print(vals.shape)
codes, vals = encode_22sgRNA_data(with_indel=True)
print(codes.shape)
print(vals.shape)
"""

"""
codes, vals = encode_CD33_mut(with_indel=True)
print(codes)
"""

# encode_cd33_indels_data()

# build_22sgRNA_data()
# encode_22sgRNA_data()

"""
import pickle as pkl
import os
file_path = "../tmp_code/CIRCLE_seq_data_for_testing_dim7_excluded_seqpairs.pkl"
if os.path.exists(file_path):
    X, y, read, sgRNA_types, infos = pkl.load(open(file_path, "rb"))
else:
    X, y, read, sgRNA_types, infos = encode_CIRCLE_data(dim_half=False)
    X = X.reshape((len(X), 1, 24, 7))
    pkl.dump([X, y, read, sgRNA_types, infos], open(file_path, "wb"))
"""








