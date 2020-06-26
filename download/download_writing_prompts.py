import os
import pickle
from tqdm import tqdm

from transformers import GPT2Tokenizer


def main():
    os.system('curl https://dl.fbaipublicfiles.com/fairseq/data/'
              'writingPrompts.tar.gz | tar xvzf -')

    os.rename('writingPrompts/valid.wp_source', 'writingPrompts/dev.wp_source')
    os.rename('writingPrompts/valid.wp_target', 'writingPrompts/dev.wp_target')

    save_dir = 'data/wp'
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    split_size = {
        'train': 10000,
        'dev': 5000,
        'test': 1000
    }

    for split in ['train', 'dev', 'test']:
        src_lines = open(f'writingPrompts/{split}.wp_source').readlines()
        tgt_lines = open(f'writingPrompts/{split}.wp_target').readlines()

        examples = []
        for src, tgt in tqdm(zip(src_lines, tgt_lines),
                             desc=split, total=len(tgt_lines)):
            src= src.strip().replace('<newline>', '\n')
            tgt = tgt.strip().replace('<newline>', '\n')

            if len(tokenizer.tokenize(
                    f'{src} [SEP] {tgt} <|endoftext|>')) > 1024:
                continue

            examples.append({
                'condition': src,
                'text': tgt
            })

            if len(examples) >= split_size[split]:
                break

        print(f'#{split}: {len(examples)}')
        pickle.dump(examples, open(f'{save_dir}/{split}.pickle', 'wb'))


if __name__ == '__main__':
    main()
