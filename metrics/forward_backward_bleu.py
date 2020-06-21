import nltk
from tqdm import trange
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import multiprocessing


def get_bleus(tokenized_ref, tokenized_hypo, n):
    f_scores = []
    for i in trange(len(tokenized_hypo), desc=f'Evaluating Forward BLEU{n}'):
        f_scores.append(sentence_bleu(
            references=tokenized_ref,
            hypothesis=tokenized_hypo[i],
            weights=[1 / n] * n,
            smoothing_function=SmoothingFunction().method1))

    b_scores = []
    for i in trange(len(tokenized_ref), desc=f'Evaluating Backward BLEU{n}'):
        b_scores.append(sentence_bleu(
            references=tokenized_hypo,
            hypothesis=tokenized_ref[i],
            weights=[1 / n] * n,
            smoothing_function=SmoothingFunction().method1))

    return get_avg(f_scores), get_avg(b_scores)


def evaluate_forward_backward_bleu(ref_texts, hypo_texts):
    tokenized_ref = [nltk.word_tokenize(text) for text in ref_texts]
    tokenized_hypo = [nltk.word_tokenize(text) for text in hypo_texts]

    results = {
        'n_ref': len(ref_texts),
        'n_hypo': len(hypo_texts)
    }

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    self_bleu_n = pool.starmap(
        get_bleus, [(tokenized_ref, tokenized_hypo, n) for n in [2, 3, 4, 5]])

    for n, (bleu_f, bleu_b) in enumerate(self_bleu_n, 2):
        results[f'forward_bleu{n}'] = bleu_f
        results[f'backward_bleu{n}'] = bleu_b
        results[f'ha_bleu{n}'] = 2. * bleu_f * bleu_b / (bleu_f + bleu_b)

    return results


def get_avg(l):
    return sum(l) / len(l)