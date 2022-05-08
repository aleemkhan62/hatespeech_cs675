import numpy as np
import argparse as ap
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf(infile):
  tweets = np.load(infile, allow_pickle=True)
  tfidf = TfidfVectorizer()
  vec = tfidf.fit_transform(tweets)
  tfidf_vec = vec.toarray()
  return tfidf_vec
  

def main(args):
  out = get_tfidf(args.input)
  np.save("tfidf_features.npy", out)
  print("tfidf features extracted at tfidf_features.npy")

if __name__ == '__main__':
  p = ap.ArgumentParser()
  p.add_argument('--input', type=str, required=True)
  args = p.parse_args()
  main(args)