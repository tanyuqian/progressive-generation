import fire
import os
import pickle

from metrics.ms_jaccard import evaluate_ms_jaccard
from metrics.frechet_bert_distance import evaluate_frechet_bert_distance
from metrics.tfidf_distance import evaluate_tfidf_distance
from metrics.forward_backward_bleu import evaluate_forward_backward_bleu


def eval_all_metrics(ref_texts, hypo_texts, label):
    os.makedirs('eval_logs/ms_jaccard', exist_ok=True)
    msj_results = evaluate_ms_jaccard(
        hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(msj_results, open(
        f'eval_logs/ms_jaccard/{label}.pickle', 'wb'))

    os.makedirs('eval_logs/tfidf_distance', exist_ok=True)
    wfd_results = evaluate_tfidf_distance(
        hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(wfd_results, open(
        f'eval_logs/tfidf_distance/{label}.pickle', 'wb'))

    os.makedirs('eval_logs/frechet_bert_distance', exist_ok=True)
    fbd_results = evaluate_frechet_bert_distance(
        hypo_texts=hypo_texts, ref_texts=ref_texts)
    pickle.dump(fbd_results, open(
        f'eval_logs/frechet_bert_distance/{label}.pickle', 'wb'))

    # slow to run, uncomment it when necessary
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

    label = f'{dataset}_first-{first_model}_{prog_steps}'
    eval_all_metrics(ref_texts, hypo_texts, label=label)


if __name__ == '__main__':
    fire.Fire(main)