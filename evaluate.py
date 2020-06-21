import fire
import os
import pickle

from metrics.ms_jaccard import evaluate_ms_jaccard
from metrics.frechet_bert_distance import evaluate_frechet_bert_distance
from metrics.word_feature_distance import evaluate_word_feature_distance
from metrics.forward_backward_bleu import evaluate_forward_backward_bleu
from metrics.gram_matrix import evaluate_gram_matrix
from metrics.ngram_similarity import evaluate_ngram_similarity
from metrics.self_bleu import evaluate_self_bleu
from metrics.bert_nsp import evaluate_bert_nsp
from metrics.sentence_repetition import evaluate_sentence_repetition
from metrics.recover import evaluate_recover


def eval_all_metrics(ref_texts, hypo_texts, label, dev_texts=None):
    os.makedirs('eval_logs/ms_jaccard', exist_ok=True)
    msj_results = evaluate_ms_jaccard(ref_texts, hypo_texts)
    pickle.dump(msj_results, open(
        f'eval_logs/ms_jaccard/{label}.pickle', 'wb'))

    if dev_texts is not None:
        msj_results = evaluate_ms_jaccard(ref_texts, dev_texts)
        pickle.dump(msj_results, open(
            f'eval_logs/ms_jaccard/dev.pickle', 'wb'))

    # os.makedirs('eval_logs/ngram_similarity', exist_ok=True)
    # ng_results = evaluate_ngram_similarity(texts=hypo_texts)
    # pickle.dump(ng_results, open(
    #     f'eval_logs/ngram_similarity/{label}.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/ngram_similarity', exist_ok=True)
    # ng_results = evaluate_ngram_similarity(texts=ref_texts)
    # pickle.dump(ng_results, open(
    #     f'eval_logs/ngram_similarity/ref.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/sentence_repetition', exist_ok=True)
    # sr_results = evaluate_sentence_repetition(texts=hypo_texts)
    # pickle.dump(sr_results, open(
    #     f'eval_logs/sentence_repetition/{label}.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/sentence_repetition', exist_ok=True)
    # sr_results = evaluate_sentence_repetition(texts=ref_texts)
    # pickle.dump(sr_results, open(
    #     f'eval_logs/sentence_repetition/ref.pickle', 'wb'))

    os.makedirs('eval_logs/word_feature_distance', exist_ok=True)
    wfd_results = evaluate_word_feature_distance(
        hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(wfd_results, open(
        f'eval_logs/word_feature_distance/{label}.pickle', 'wb'))

    if dev_texts is not None:
        wfd_results = evaluate_word_feature_distance(
            hypo_texts=hypo_texts, ref_texts=dev_texts)
        pickle.dump(wfd_results, open(
            f'eval_logs/word_feature_distance/dev.pickle', 'wb'))

    os.makedirs('eval_logs/frechet_bert_distance', exist_ok=True)
    fbd_results = evaluate_frechet_bert_distance(
        hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(fbd_results, open(
        f'eval_logs/frechet_bert_distance/{label}.pickle', 'wb'))

    if dev_texts is not None:
        fbd_results = evaluate_frechet_bert_distance(
            hypo_texts=hypo_texts, ref_texts=dev_texts)
        pickle.dump(fbd_results, open(
            f'eval_logs/frechet_bert_distance/dev.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/bert_nsp', exist_ok=True)
    # bnsp_results = evaluate_bert_nsp(texts=hypo_texts)
    # pickle.dump(bnsp_results, open(
    #     f'eval_logs/bert_nsp/{label}.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/bert_nsp', exist_ok=True)
    # bnsp_results = evaluate_bert_nsp(texts=ref_texts)
    # pickle.dump(bnsp_results, open(
    #     f'eval_logs/bert_nsp/ref.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/gram_matrix', exist_ok=True)
    # gm_results = evaluate_gram_matrix(
    #     ref_texts=ref_texts, hypo_texts=hypo_texts)
    # pickle.dump(gm_results, open(
    #     f'eval_logs/gram_matrix/{label}.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/self_bleu', exist_ok=True)
    # sb_results = evaluate_self_bleu(texts=hypo_texts)
    # pickle.dump(sb_results, open(
    #     f'eval_logs/self_bleu/{label}.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/self_bleu', exist_ok=True)
    # sb_results = evaluate_self_bleu(texts=ref_texts)
    # pickle.dump(sb_results, open(
    #     f'eval_logs/self_bleu/ref.pickle', 'wb'))
    #
    # os.makedirs('eval_logs/forward_backward_bleu', exist_ok=True)
    # bleu_results = evaluate_forward_backward_bleu(
    #     hypo_texts=hypo_texts, ref_texts=ref_texts)
    # pickle.dump(bleu_results, open(
    #     f'eval_logs/forward_backward_bleu/{label}.pickle', 'wb'))


def main(dataset,
         first_model,
         prog_steps,
         top_k=-1,
         top_p=0.95):

    prog_vocabs = prog_steps.split('-')
    assert prog_vocabs[0] == 'null' and prog_vocabs[-1] == 'full'

    decoding = 'top_'
    if top_k > 0:
        decoding += f'k{top_k}'
    if top_p > 0:
        decoding += f'p{top_p}'

    test_examples = pickle.load(open(f'data/{dataset}/test.pickle', 'rb'))
    ref_texts = [example['text'] for example in test_examples]

    gen_dir = f'generated_texts/' \
              f'{dataset}_first-{first_model}_{prog_steps}/{decoding}'

    hypo_texts = []
    for example in pickle.load(open(f'{gen_dir}/gen.pickle', 'rb')):
        hypo_texts.append(example['prog_gens'][-1])

    label = f'first-{first_model}_' + prog_steps
    eval_all_metrics(ref_texts, hypo_texts, label=label)


if __name__ == '__main__':
    fire.Fire(main)