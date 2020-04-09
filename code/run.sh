# run CRISPR-Net to score gRNA-target pairs
python /code/CRISPR_Net.py /code/input_examples/on_off_seq_pairs.txt

# run CRISPR-Net-Aggregate to produce a overall off-target score for a gRNA
python /code/CRISPR_Net_Aggregate.py /code/input_examples/aggregate_example_GACCTTGCATTGTACCCGAG.csv

# evaluate CRISPR-Net based on the dataset in /data
python /code/evaluate_CRISPR_Net.py