# CRISPR-Net: A Fused Recurrent Convolutional Network Quantifies CRISPR Off-target Activities with Indels and Mismatches
This repository includes a deep convolutional neural network for predicting the off-targets in CRISPR-Cas9 gene editing. 

# PREREQUISITE
The models for off-target predicitons were conducted by using Python 2.7.13 and TensorFLow v1.4.1. 
Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>scikit-learn</p></li>
<li><p>TensorFlow</p></li>
</ul>

The Keras version of CNN were conducted by Python 3.6, TensorFlow 1.9.0, Keras 2.2.0.
Following Python packages should be installed:
<ul>
<li><p>scipy</p></li>
<li><p>numpy</p></li>
<li><p>pandas</p></li>
<li><p>Keras</p></li>
<li><p>TensorFlow</p></li>
</ul>

# REFERENCE

[1] Hui Peng, Yi Zheng, Zhixun Zhao, Tao Liu, Jinyan Li; Recognition of CRISPR/Cas9 off-target sites through ensemble learning of uneven mismatch distributions, Bioinformatics, Volume 34, Issue 17, 1 September 2018, Pages i757–i765, https://doi.org/10.1093/bioinformatics/bty558

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
