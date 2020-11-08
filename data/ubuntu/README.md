## Ubuntu-v2 retrieval dataset:
https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip?dl=0


This .zip file includes the datasets (training/testint/validation) used in the experiments of paper:
Incorporating Loose-Structured Knowledge into LSTM with Recall-Gate for Conversation Modeling.

The datasets are extracted from the corpus: http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ 
Negtive sampling is conducted to produce balanced training set and 1:9 validation/testing sets following the paper of Lowe et al. (2015)

The details of the datasets are give below:
1. train.txt: 1 million training samples (pos:neg=1:1)
2. valid.txt: 50,000 samples for validation (pos:neg=1:9)
3. test.txt: 50,000 samples for testing (pos:neg=1:9)
4. vocab.txt: Vocabulary of the datasets. 

