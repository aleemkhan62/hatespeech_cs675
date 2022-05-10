import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel
from transformers import TFAutoModel
from transformers import AutoTokenizer
import argparse as ap
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

def get_classifier(cls):
  if cls == 'logistic-regression':
    return LogisticRegression(random_state=0)
  elif cls == 'svm':
    return make_pipeline(StandardScaler(), SVC(gamma='auto'))
  elif cls == 'mlp':
    return MLPClassifier(verbose=True, hidden_layer_sizes=(64, 64, 64, 3))
  else:
    raise ValueError(f"invalid classifier {cls}")

def main(args):
  clf = get_classifier(args.classifier)
  data_dir = "/exp/akhan/labeled_data.csv"
  df = pd.read_csv(data_dir)
  tweets = df['tweet']
  X = np.load(args.input_features)
  y = np.load(args.input_labels)
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
  clf.fit(X_train, y_train)
  res = clf.predict(X_val)
  print(f"accuracy: {accuracy_score(y_val, res)}")
  f1score= f1_score(y_val, res, average='weighted')
  print(f"f1 score: {f1score}")

if __name__ == '__main__':
  p = ap.ArgumentParser()
  p.add_argument('--classifier',
                    default='logistic-regression',
                    const='logistic-regression',
                    nargs='?',
                    choices=['logistic-regression', 'svm', 'mlp'],
                    help='Classifier to run (default: %(default)s)')
  p.add_argument('--input-features', type=str, required=True)
  p.add_argument('--input-labels', type=str, required=True)
  args = p.parse_args()
  main(args)
