import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
from transformers import TFAutoModel
from transformers import AutoTokenizer
import numpy as np
import argparse as ap
import pandas as pd
from tqdm import tqdm

def extract_bert_feats(m, tok, tweets, batch_size=64):
  batch_docs = []
  
  document_features = []
  print(f"Extracting features w/ BERT, batch size = {batch_size}")
  for idx, tweet in enumerate(tweets):
    batch_docs.append(tweet)
    if len(batch_docs) >= batch_size or idx == len(tweets)-1:
      tokenized = tok(batch_docs, return_tensors='tf', padding=True, truncation=True)
      feats = m(tokenized, training=False).pooler_output
      document_features.append(feats)
      batch_docs = []
  return np.concatenate(document_features, axis=0)
  
def main(args):
  print(f"Note that feature extraction can take up to 1 hour on a CPU and < 1 min on a GPU. TF will use a GPU if available")
  if args.bert_type == 'base':
    m = TFBertModel.from_pretrained('bert-base-uncased')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
  elif args.bert_type == 'tweet':
    tok = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    m = TFAutoModel.from_pretrained('vinai/bertweet-base')
  else:
    raise ValueError(f"Invalid bert model {args.bert_type}")
  tweets = np.load(args.input_file, allow_pickle=True)
  bert_feats = extract_bert_feats(m, tok, tweets)
  np.save(f'bert_{args.bert_type}_feats.npy', bert_feats)

if __name__ == '___main__':
  p = ap.ArgumentParser()
  p.add_argument('--input-file', type=str, required=True)
  p.add_argument('--bert-type', type=str, default='base')
  args = p.parse_args()
  main(args)
