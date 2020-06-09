import os
import fire
import pickle

from models.bart import BART

BATCH_SIZE = 5
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1


def get_vocab(dataset, vocab_size):
    if vocab_size == 'null':
        return None
    return pickle.load(open(f'data/{dataset}/vocab_{vocab_size}.pickle', 'rb'))


def load_data(dataset, split, vocab_size, keep_condition):
    if vocab_size in ['null', 'full']:
        examples = pickle.load(open(f'data/{dataset}/{split}.pickle', 'rb'))
        if vocab_size == 'null':
            return [example['condition'] for example in examples]
        else:
            return [example['text'] for example in examples]
    else:
        examples = pickle.load(open(
            f'data/{dataset}/extracted_{split}_{vocab_size}words.pickle', 'rb'))
        if keep_condition:
            return [example['condition'] + ' [SEP] ' + example['extracted_text']
                    for example in examples]
        else:
            return [example['extracted_text'] for example in examples]


def main(vocab_size_src='null',
         vocab_size_tgt='full',
         dataset='writing_prompts',
         n_epochs=3):

    if os.path.exists(f'bart_{vocab_size_src}-{vocab_size_tgt}_training_logs'):
        return

    bart = BART()

    for split in ['train', 'dev']:
        src_texts = load_data(
            dataset, split, vocab_size_src, keep_condition=True)
        tgt_texts = load_data(
            dataset, split, vocab_size_tgt, keep_condition=False)

        bart.load_data(set_type=split, src_texts=src_texts, tgt_texts=tgt_texts)

    train_steps = n_epochs * (len(bart.dataset['train']) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    bart.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    bart.create_training_log(
        eval_steps=len(bart.dataset['train']) // BATCH_SIZE,
        label=f'bart_{dataset}_{vocab_size_src}-{vocab_size_tgt}')

    noise_vocab = get_vocab(dataset, vocab_size_src)
    for epoch in range(n_epochs):
        bart.train_epoch(batch_size=BATCH_SIZE, noise_vocab=noise_vocab)


if __name__ == '__main__':
    fire.Fire(main)
