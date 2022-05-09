import numpy as np
import argparse as ap
from gensim.models import Word2Vec

def get_word2vec(infile):
    vec_size = 100
    tweets = np.load(infile, allow_pickle=True)
    W2Vdata = []
    for sentence in tweets:
        tokens = list(sentence.lower().split())
        W2Vdata.append(tokens)
    model = Word2Vec(sentences=W2Vdata, vector_size=vec_size, window=3, min_count=3, workers=4, sg=1, negative=5)
    word_vectors = model.wv
    #process tweets to replace low-count words
    lowcountwords = []
    for i,row in enumerate(W2Vdata):
        for j,token in enumerate(row):
            if token not in word_vectors.index_to_key:
                W2Vdata[i][j] = "<UNK>"
                if token not in lowcountwords:
                    lowcountwords.append(token)
    #rerun to include "UNK" to represent low-count words
    model = Word2Vec(sentences=W2Vdata, vector_size=vec_size, window=3, min_count=3, workers=4, sg=1, negative=5)
    word2vec = model.wv
    print(len(lowcountwords))
    #get tweet-level representation
    tweet2vec = np.zeros([len(W2Vdata),vec_size])
    for i,sentence in enumerate(W2Vdata):
        sentence = np.unique(sentence)
        if len(sentence) > 1:
            tweet_vector = np.mean(word2vec[sentence], axis=0)
        else:
            tweet_vector = np.zeros(vec_size)
        tweet2vec[i,:] = tweet_vector
    print(tweet2vec[1])
    return word2vec, tweet2vec

def main(args):
    word2vec, tweet2vec = get_word2vec(args.input)
    word2vec.save("word2vec_features.kv")
    np.save("tweet2vec_features.npy", tweet2vec)
    print("word2vec word-level features extracted at word2vec_features.kv")
    print("word2vec tweet-level features extracted at tweet2vec_features.npy")
    

if __name__ == '__main__':
    p = ap.ArgumentParser()
    p.add_argument('--input', type=str, required=True)
    args = p.parse_args()
    main(args)