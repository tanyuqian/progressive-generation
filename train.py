import fire
import os


def main(dataset, prog_steps, first_model):
    prog_vocabs = prog_steps.split('-')
    assert prog_vocabs[0] == 'null' and prog_vocabs[-1] == 'full'

    for vocab in prog_steps[1:-1]:
        os.system(f'python make_stage.py --dataset {dataset} --rate {vocab}')

    if first_model == 'bart':
        os.system(f'python bart_finetune.py --dataset {dataset} '
                  f'--src_vocab null --tgt_vocab {prog_vocabs[1]}')
    else:
        os.system(f'python gpt2_finetune.py --dataset {dataset} '
                  f'--vocab {prog_vocabs[1]} --gpt2_type {first_model}')


if __name__ == '__main__':
    fire.Fire(main)
