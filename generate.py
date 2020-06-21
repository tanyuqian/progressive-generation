import fire
import os
from tqdm import tqdm
import pickle

from models.bart import BART
from models.gpt2 import GPT2


def get_model_list(dataset, prog_vocabs, first_model):
    model_list = []
    for i in range(len(prog_vocabs) - 1):
        if i == 0 and 'gpt2' in first_model:
            model = GPT2(gpt2_type=first_model)
            model.load_model(f'training_logs/{first_model}_{dataset}_'
                             f'{prog_vocabs[1]}words/best_model.pt')
        else:
            model = BART()
            model.load_model(f'training_logs/bart_{dataset}_{prog_vocabs[i]}-'
                             f'{prog_vocabs[i+1]}/best_model.pt')

        model_list.append(model)

    return model_list


def main(dataset,
         prog_steps,
         first_model,
         top_k=-1,
         top_p=0.95):

    prog_vocabs = prog_steps.split('-')
    assert prog_vocabs[0] == 'null' and prog_vocabs[-1] == 'full'

    model_list = get_model_list(dataset, prog_vocabs, first_model)

    decoding = 'top_'
    if top_k > 0:
        decoding += f'k{top_k}'
    if top_p > 0:
        decoding += f'p{top_p}'

    test_examples = pickle.load(open(f'data/{dataset}/test.pickle', 'rb'))

    gen_dir = f'generated_texts/{dataset}_first-{first_model}_{prog_steps}/' \
              f'{decoding}'
    os.makedirs(gen_dir, exist_ok=True)

    log_file = open(f'{gen_dir}/gen.txt', 'w')
    gens = []
    for example in tqdm(test_examples, desc='Generating'):
        condition, truth = example['condition'], example['text']

        if 'gpt2' in first_model:
            prog_gens = [model_list[0].generate(
                cond=condition, top_k=top_k, top_p=top_p)]
        else:
            prog_gens = [model_list[0].generate(
                src_text=condition, top_k=top_k, top_p=top_p)]

        for model in model_list[1:]:
            prog_gens.append(model.generate(
                src_text=condition + ' [SEP] ' + prog_gens[-1],
                top_k=top_k, top_p=top_p))

        gens.append({
            'condition': condition,
            'truth': truth,
            'prog_gens': prog_gens,
            'top_k': top_k,
            'top_p': top_p
        })

        print(f'CONDITION:\n{condition}\n', '-' * 50, '\n\n',
              f'TRUTH:\n{truth}\n', '=' * 100, '\n\n', file=log_file)
        for step, text in enumerate(prog_gens):
            print(f'STEP_{step}:\n{text}\n', '-' * 50, '\n\n', file=log_file)
        print('=' * 50, file=log_file)
        log_file.flush()

    pickle.dump(gens, open(f'{gen_dir}/gen.pickle', 'wb'))


if __name__ == '__main__':
    fire.Fire(main)
