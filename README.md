# CRISPR-Net: A Fused Recurrent Convolutional Network Quantifies CRISPR Off-target Activities with Indels and Mismatches
This repository includes a fused long-term recurrent convolutional neural network for predicting the off-targets activities with indels and mismatches in CRISPR/Cas9 gene editing.

# PREREQUISITE
The models for off-target predicitons were conducted by using Python 2.7.13 and TensorFlow v1.4.1. 
Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>scikit-learn</p></li>
<li><p>TensorFlow</p></li>
</ul>

# Usage

CRISPR-Net can run with:

    CRISPR_net.py [-h] input_file

positional arguments: input_file

optional arguments:
  -h, --help  show this help message and exit

Example input file:

GAGT_CCGAGCAGAAGAAGAATGG,GAGTACCAAGTAGAAGAAAAATTT
GTTGCCCCACAGGGCAGTAAAGG,GTGGACACCCCGGGCAGGAAAGG
GGGTGGGGGGAGTTTGCTCCCGG,GTGTGGGGTAAATTTGCTCCCAG
GGGTGGGGGGAGTTTGCTCCAGG,AGGTGGGGTGA_TTTGCTCCAGG

Save it as 'input.txt'.

Now you can run CRISPR-Net as following:

    $> ./CRISPR_net.py input.txt
    ...
  
--------------------------------------------------
CRISPR-Net-mismatches-only can run with:

    CRISPR_net_mismatches_only.py [-h] input_file

positional arguments: input_file

Example input file:

GATGGTAGATGGAGACTCAGAGG,GGTAGGAAATGGAGAGGCAGAGG
GGGTGGGGGGAGTTTGCTCCCGG,GTGTGGGGTAAATTTGCTCCCAG

optional arguments:
  -h, --help  show this help message and exit


# CONTAINS:
<ul>
<li><p>CFDScoring/cfd-score-calculator.py : Python script to run CFD score </p></li>
<li><p>predictions/cnn_std_prediction_TF.py : CNN_std conducted by TensorFlow</p></li>
<li><p>predictions/cnn_std_keras.py : CNN_std conducted by Keras used for off-target prediction </p></li>
</p></li>
</ul>

---------------------------------------
Jiecong Lin

jieconlin3-c@my.cityu.edu.hk

January 10 2019
