import os
import fire
import pickle

from models.gpt2 import GPT2


BATCH_SIZE = 5
LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1


def load_data(dataset, split, vocab_size):
    if vocab_size == 'full':
        examples = pickle.load(open(f'data/{dataset}/{split}.pickle', 'rb'))
        return [example['condition'] for example in examples], \
               [example['text'] for example in examples]
    else:
        examples = pickle.load(open(
            f'data/{dataset}/extracted_{split}_{vocab_size}words.pickle', 'rb'))
        return [example['condition'] for example in examples],\
               [example['text'] for example in examples]


def main(dataset='wp', vocab_size='full', gpt2_type='gpt2', n_epochs=3):
    if os.path.exists(f'{gpt2_type}_{dataset}_{vocab_size}words_training_logs'):
        return

    gpt2 = GPT2(gpt2_type=gpt2_type)

    for split in ['train', 'dev']:
        conds, texts = load_data(dataset, split, vocab_size)
        gpt2.load_data(split=split, conds=conds, texts=texts)

    train_steps = n_epochs * (len(gpt2.train_dataset) // BATCH_SIZE + 1)
    warmup_steps = int(train_steps * WARMUP_PROPORTION)
    gpt2.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    gpt2.creat_log_dir(
        eval_steps=len(gpt2.train_dataset) // BATCH_SIZE,
        label=f'{gpt2_type}_{dataset}_{vocab_size}words')

    for epoch in range(n_epochs):
        gpt2.train_epoch(batch_size=BATCH_SIZE)


if __name__ == '__main__':
    fire.Fire(main)
