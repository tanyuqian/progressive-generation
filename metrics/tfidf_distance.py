import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_vocab(texts):
    vect = CountVectorizer(min_df=5)
    vect.fit(texts)

    return vect.get_feature_names()


def get_feature(texts, vocab):
    vect = TfidfVectorizer(vocabulary=vocab)
    features = vect.fit_transform(texts)

    avg_feature = np.zeros(features.shape[1])
    for i in range(features.shape[0]):
        n_sent = len(nltk.sent_tokenize(texts[i]))
        for idx, data in zip(features[i].indices, features[i].data):
            avg_feature[idx] += data / n_sent

    return avg_feature / features.shape[0]


def evaluate_tfidf_distance(ref_texts, hypo_texts):
    print('Evaluating TF-IDF Distance...')

    vocab = get_vocab(ref_texts)

    results = {
        'n_ref': len(ref_texts),
        'n_hypo': len(hypo_texts)
    }

    ref_feature = get_feature(ref_texts, vocab)
    hypo_feature = get_feature(hypo_texts, vocab)
    results[f'tfidf_distance'] = np.linalg.norm(ref_feature - hypo_feature)

    return results