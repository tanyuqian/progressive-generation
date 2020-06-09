import fire
import os


def main(dataset, prog_steps, first_model):
    prog_vocabs = prog_steps.split('-')
    assert prog_vocabs[0] == 'null' and prog_vocabs[-1] == 'full'

    for vocab in prog_vocabs[1:-1]:
        os.system(f'python make_stage.py '
                  f'--dataset {dataset} '
                  f'--rate {vocab}')

    if first_model == 'bart':
        os.system(f'python bart_finetune.py '
                  f'--dataset {dataset} '
                  f'--src_vocab null '
                  f'--tgt_vocab {prog_vocabs[1]}')
    else:
        os.system(f'python gpt2_finetune.py '
                  f'--dataset {dataset} '
                  f'--vocab {prog_vocabs[1]} '
                  f'--gpt2_type {first_model}')

    for i in range(1, len(prog_steps)):
        os.system(f'python bart_finetune.py --dataset {dataset} '
                  f'--src_vocab {prog_vocabs[i]} '
                  f'--tgt_vocab {prog_vocabs[i+1]}')


if __name__ == '__main__':
    fire.Fire(main)
