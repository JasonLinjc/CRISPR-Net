# -*- coding: utf-8 -*-
# @Time     :10/15/18 9:50 PM
# @Auther   :Jason Lin
# @File     :Encoder_sgRNA_off$.py
# @Software :PyCharm

import numpy as np

class Encoder:

    def __init__(self, sgRNA_seq, off_seq, with_indel = False, with_category = False, label = None, with_reg_val = False, value = None, dim_half=True):
        self.sgRNA_seq = sgRNA_seq
        self.off_seq = off_seq
        self.indel_flag = with_indel
        self.encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}

        if with_category:
            self.label = label

        if with_reg_val:
            self.value = value

        if dim_half:
            self.encode_sgRNA_off_old()
        else:
            self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        if self.indel_flag:
            encoded_dict = self.encoded_dict_indel
        else:
            encoded_dict = self.encoded_dict

        sgRNA_bases = list(self.sgRNA_seq)
        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        if self.indel_flag:
            encoded_dict = self.encoded_dict_indel
        else:
            encoded_dict = self.encoded_dict
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_sgRNA_off_old(self):
        self.encode_sgRNA()
        self.encode_off()
        sgRNA_off_code = []
        for i in range(len(self.sgRNA_code)):
            sgRNA_off_code.append(np.bitwise_or(self.sgRNA_code[i], self.off_code[i]))
        self.sgRNA_off_code = np.array(sgRNA_off_code)

    def encode_on_off_dim7(self):
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.sgRNA_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)


"""
e = Encoder(sgRNA_seq="AGCTGN", off_seq="CGCGTT", dim_half=False)
print(e.on_off_code)
"""










