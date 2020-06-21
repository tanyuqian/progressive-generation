from tqdm import tqdm

import torch
import numpy as np
from scipy import linalg

from transformers import BertTokenizer, BertModel


BERT_MODEL = 'bert-large-cased'
BERT_MAX_LENGTH = 512


def evaluate_frechet_bert_distance(hypo_texts, ref_texts):
    print('Evaluating Frechet Bert Distance...')

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=False)
    bert = BertModel.from_pretrained(
        BERT_MODEL, output_hidden_states=True).to('cuda')

    features = get_features(hypo_texts, tokenizer, bert)
    ref_features = get_features(ref_texts, tokenizer, bert)

    results = {
        'n_hypo': len(hypo_texts),
        'n_ref': len(ref_texts)
    }

    dist = []
    for layer in range(bert.config.num_hidden_layers + 1):
        mu, sigma = get_mu_sigma(features[layer])
        ref_mu, ref_sigma = get_mu_sigma(ref_features[layer])

        dist.append(np.sqrt(
            calculate_frechet_distance(ref_mu, ref_sigma, mu, sigma)))

    assert dist[0] < 1e-5

    for l, r in [(1, 8), (9, 16), (17, 24)]:
        results[f'fbd_{l}-{r}'] = sum(dist[l:r + 1])

    return results


def get_features(texts, tokenizer, bert):
    all_features = [[] for _ in range(bert.config.num_hidden_layers + 1)]
    for text in tqdm(texts):
        input_ids = tokenizer.encode(text, max_length=BERT_MAX_LENGTH)

        with torch.no_grad():
            outputs = bert(torch.tensor([input_ids]).to('cuda'))

        for layer in range(bert.config.num_hidden_layers + 1):
            all_features[layer].append(outputs[2][layer][0][0].tolist())

    for layer in range(bert.config.num_hidden_layers + 1):
        all_features[layer] = np.array(all_features[layer])

    return all_features


def get_mu_sigma(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


# https://arxiv.org/pdf/1706.08500.pdf
# from https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
