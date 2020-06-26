import nltk
from scipy.stats.mstats import gmean


def evaluate_ms_jaccard(ref_texts, hypo_texts):
    print('Evaluating MS-Jaccard...')

    n_sents_ref = get_cnt_sents(ref_texts)
    n_sents_hypo = get_cnt_sents(hypo_texts)

    scores = [None]
    for gram_n in [1, 2, 3, 4, 5]:
        ngram_cnt_ref = get_ngram_cnt(ref_texts, gram_n)
        ngram_cnt_hypo = get_ngram_cnt(hypo_texts, gram_n)

        min_sum, max_sum = 0, 0
        for ngram in set(
                list(ngram_cnt_ref.keys()) + list(ngram_cnt_hypo.keys())):
            cnt_ref = ngram_cnt_ref.get(ngram, 0) / n_sents_ref
            cnt_hypo = ngram_cnt_hypo.get(ngram, 0) / n_sents_hypo

            min_sum += min(cnt_ref, cnt_hypo)
            max_sum += max(cnt_ref, cnt_hypo)
        scores.append(min_sum / max_sum)

    results = {
        'n_ref': len(ref_texts),
        'n_hypo': len(hypo_texts)
    }

    for gram_n in [2, 3, 4, 5]:
        results[f'ms_jaccard{gram_n}'] = gmean(scores[1:gram_n + 1])

    return results


def get_ngram_cnt(texts, gram_n):
    ngram_cnt = {}
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())

        for i in range(len(tokens) - gram_n):
            ngram = ' '.join(tokens[i: i + gram_n])
            if ngram not in ngram_cnt:
                ngram_cnt[ngram] = 0
            ngram_cnt[ngram] += 1

    return ngram_cnt


def get_cnt_sents(texts):
    cnt_all_sent = 0
    for text in texts:
        cnt_all_sent += len(nltk.sent_tokenize(text))
    return cnt_all_sent
