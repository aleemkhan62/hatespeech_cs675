# hatespeech_cs675
Repo for cs475/675 course project
## Feature Extraction

### TF-IDF
To extract tfidf features, run the following line:
python3 tfidf.py --input path_to_preprocessed_input_data

### word2vec
To extract word2vec features, run the following line:
python3 word2vec.py --input path_to_preprocessed_input_data
Tested on Gensim 4.1.2.

### Transformers / BERT
To extract BERT features, run the following line:
python3 bert.py --input-file path_to_preprocessed_input_data --bert-type {base|tweet}
Note: Feature extaction for the entire dataset can take up to 1 hour on a cpu.


## Downstream classification
We've simplified our code for reproducability, after running the above feature extractors, use the run_classifiers.py script to obtain scores. The input arguments are the outputs of the above feature extractor scripts.
python3 run_classifiers.py --classifier classifier-type --input-features path_to_features --input-labels path_to_labels
