# CRISPR-Net: A Recurrent Convolutional Network Quantifies CRISPR Off-target Activities with Indels and Mismatches
This repository includes a recurrent convolutional neural network named CRISPR-Net for predicting the off-targets activities with insertions, deletions and mismatches in CRISPR/Cas9 gene editing. There are two command-line tools that can be used to quantify the off-target activities induced by CRISPR guide RNA. One is **CRISPR_Net.py** which can predict the off-target activities with indels and mismatches, the other is **CRISPR_Net_Aggregate.py** which aggregates the gRNA-target scores from CRISPR-Net into a single consensus off-target score.

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

    python CRISPR_net.py [-h] input_file

positional arguments: input_file

optional arguments:
  -h, --help  show the help message and exit

Example input file:

    GAGT_CCGAGCAGAAGAAGAATGG,GAGTACCAAGTAGAAGAAAAATTT
    GTTGCCCCACAGGGCAGTAAAGG,GTGGACACCCCGGGCAGGAAAGG
    GGGTGGGGGGAGTTTGCTCCCGG,GTGTGGGGTAAATTTGCTCCCAG
    GGGTGGGGGGAGTTTGCTCCAGG,AGGTGGGGTGA_TTTGCTCCAGG
    GATGGTAGATGGAGACTCAGNGG,GGTAGGAAATGGAGAGGCAGAGG
    GAGTCCGAGCAGAAGAAGAAAGG,GAGTTAGAGCAGAAGAAGAAAGG

Save it as 'input.txt'.

Now you can run CRISPR-Net as following:

    $> python ./CRISPR_net.py input.txt
    ...
    
Then output will be generated:

- The first column of the output file indicates the on-target sequence,
- The second column of the output file indicates the off-target sequence,
- The third column is the off-taget score predicted by CRISPR-Net

and saved to ./CRISPR_net_results.csv:


    on_seq,off_seq,CRISPR_Net_score
    GAGT_CCGAGCAGAAGAAGAATGG,GAGTACCAAGTAGAAGAAAAATTT,0.000000e+00
    GTTGCCCCACAGGGCAGTAAAGG,GTGGACACCCCGGGCAGGAAAGG,1.490116e-07
    GGGTGGGGGGAGTTTGCTCCCGG,GTGTGGGGTAAATTTGCTCCCAG,6.078471e-01
    GGGTGGGGGGAGTTTGCTCCAGG,AGGTGGGGTGA_TTTGCTCCAGG,3.451956e-01
    GATGGTAGATGGAGACTCAGNGG,GGTAGGAAATGGAGAGGCAGAGG,5.029583e-05
    GAGTCCGAGCAGAAGAAGAAAGG,GAGTTAGAGCAGAAGAAGAAAGG,9.269089e-01

--------------------------------------------------

CRISPR-Net-aggregate can run with:

    python CRISPR_net_aggregate.py gRNA_offTargets.csv

positional arguments: gRNA_offTargets.csv

optional arguments:
  -h, --help show the help message and exit

Example input file:


    off_target,Gene_mark,on_target
    GAACTAGCCTTGTATCCCAGGGA,RP4-669L17.10,GACCTTGCATTGTACCCGAGGGG
    GAACTAGCCTTGTATCCCAGGGA,RP5-857K21.4,GACCTTGCATTGTACCCGAGGGG
    GTGTTTGCAATGTACCCGTGTTG,TTLL10-AS1,6,1173076,GACCTTGCATTGTACCCGAGTGG
    GACCTGTGGTTGTTCCTGAGAGG,,GACCTTGCATTGTACCCGAGAGG
    GCCCTTGGATTGGCCGCGAGGGC,,GACCTTGCATTGTACCCGAGGGG

Save it as 'gRNA_offTargets.csv'.

Now you can run CRISPR-Net-Aggregate as following:

    $> python ./CRISPR_Net_aggregate.py gRNA_offTargets.csv
    ...
    
Then off-target aggregate score will be generated. 

You can try CRISPR-Net-Aggregate on the our example file (input_example/aggregate_example_GACCTTGCATTGTACCCGAG.csv) by:

    python CRISPR_net_aggregate.py ./input_example/aggregate_example_GACCTTGCATTGTACCCGAG.csv


# CONTAINS:
<ul>
<li><p>code/CRISPR_Net.py : Python script to run CRISPR-Net for predicting off-target activities with indels and mismathces </p></li>
<li><p>code/CRISPR_Net_Aggregate.py : Python script to run CRISPR-Net-Aggregate that aggregates the gRNA-target scores from CRISPR-Net into a single consensus off-target score.</p></li>
</p></li>
<li><p>code/evaluate_CRISPR_Net.py : Python script to retrain CRISPR-Net on any dataset in /data and evaluate its performance.</p></li>
</p></li>
</ul>

