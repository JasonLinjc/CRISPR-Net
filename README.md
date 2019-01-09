# CRISPR-Net: A Fused Recurrent Convolutional Network Quantifies CRISPR Off-target Activities with Indels and Mismatches
This repository includes a fused long-term recurrent convolutional neural network named CRISPR-Net for predicting the off-targets activities with insertions, deletions and mismatches in CRISPR/Cas9 gene editing. There are two models that can be used to quantify the 

# PREREQUISITE
CRISPR-Net was conducted by Python 3.6 and Keras 2.2.4 (using TensorFlow 1.12 backend) 

Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>scikit-learn</p></li>
<li><p>TensorFlow</p></li>
<li><p>Keras</p></li>
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
    
Then output will be generated:

- The first column of the output file indicates the on-target sequence,
- The second column of the output file indicates the off-target sequence,
- The third column is the off-taget score predicted by CRISPR-Net

and saved to ./CRISPR_net_results.csv:

                       on_seq                   off_seq     CRISPR_net_score
     GAGT_CCGAGCAGAAGAAGAATGG  GAGTACCAAGTAGAAGAAAAATTT         7.498138e-10
      GTTGCCCCACAGGGCAGTAAAGG   GTGGACACCCCGGGCAGGAAAGG         8.658309e-08
      GGGTGGGGGGAGTTTGCTCCCGG   GTGTGGGGTAAATTTGCTCCCAG         6.078470e-01
      GGGTGGGGGGAGTTTGCTCCAGG   AGGTGGGGTGA_TTTGCTCCAGG         3.451956e-01
                     
--------------------------------------------------

CRISPR-Net-mismatches-only can run with:

    CRISPR_net_mismatches_only.py [-h] input_file

positional arguments: input_file

Example input file:

    GATGGTAGATGGAGACTCAGAGG,GGTAGGAAATGGAGAGGCAGAGG
    GGGTGGGGGGAGTTTGCTCCCGG,GTGTGGGGTAAATTTGCTCCCAG

Save it as 'input.txt'.

Now you can run CRISPR-Net as following:

    $> ./CRISPR_net_mismatches_only.py input.txt
    ...
    
Then output file will be generated:

- The first column of the output file indicates the on-target sequence,
- The second column of the output file indicates the off-target sequence,
- The third column is the off-taget score by CRISPR-Net-mismatches-only

and saved to ./CRISPR_net_results_mismatches_only.csv:
                     
                      on_seq                  off_seq   CRISPR_net_score
     GATGGTAGATGGAGACTCAGNGG  GGTAGGAAATGGAGAGGCAGAGG       1.283432e-17
     GAGTCCGAGCAGAAGAAGAAAGG  GAGTTAGAGCAGAAGAAGAAAGG       3.380741e-01


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
