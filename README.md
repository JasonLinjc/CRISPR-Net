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

Cas-OFFinder can run with:
  
    cas-offinder {input_file} {G|C|A}[device_id(s)] {output_file}

G stands for using GPU devices, C for using CPUs, and A for using accelerators.

(Optionally, you can set device ID in addition to G/C/A to limit number of devices used by Cas-OFFinder)

A short example may be helpful!

First, download any target organism's chromosome FASTA files. You can find one in below links:

- http://hgdownload.soe.ucsc.edu/downloads.html (UCSC genome sequences library)
- http://ensembl.org/info/data/ftp/index.html (Ensembl sequence library)

Extract all FASTA files in a directory.

For example (human chromosomes, in POSIX environment):
    
    $> wget http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz
    $> mkdir -p /var/chromosome/human_hg19
    $> tar zxf chromFa.tar.gz -C /var/chromosome/human_hg19
    $> ls -al /var/chromosome/human_hg19
      drwxrwxr-x.  2 user group      4096 2013-10-18 11:49 .
      drwxrwxr-x. 16 user group      4096 2013-11-12 12:44 ..
      -rw-rw-r--.  1 user group 254235640 2009-03-21 00:58 chr1.fa
      -rw-rw-r--.  1 user group 138245449 2009-03-21 01:00 chr10.fa
      -rw-rw-r--.  1 user group 137706654 2009-03-21 01:00 chr11.fa
      -rw-rw-r--.  1 user group 136528940 2009-03-21 01:01 chr12.fa
      -rw-rw-r--.  1 user group 117473283 2009-03-21 01:01 chr13.fa
      -rw-rw-r--.  1 user group 109496538 2009-03-21 01:01 chr14.fa
      ...
     
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
