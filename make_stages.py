import pickle
import fire
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer


def get_texts(dataset):
    conds, texts = {}, {}
    for split in ['train', 'dev', 'test']:
        print(f'loading {split} set...')
        examples = pickle.load(open(
            f'data/{dataset}/{split}.pickle', 'rb'))

        conds[split], texts[split] = [], []
        for example in examples:
            conds[split].append(example['condition'])
            texts[split].append(example['text'])

    return conds, texts


def get_vocab(texts, rate):
    vectorizer = TfidfVectorizer(min_df=5, stop_words='english')
    features = vectorizer.fit_transform(texts).tocsc()

    vocab = vectorizer.get_feature_names()
    analyzer = vectorizer.build_analyzer()

    df = 1. / np.exp(vectorizer.idf_ - 1) * (len(texts) + 1) - 1

    word_value_list = []
    for i, word in enumerate(vocab):
        assert len(features[:, i].data) == int(round(df[i]))
        word_value_list.append(
            [word, np.mean(features[:, i].data), len(features[:, i].data)])
    word_value_list.sort(key=lambda t: t[1], reverse=True)

    total = sum([len(analyzer(text)) for text in texts])
    word_counter = {word: 0 for word in vocab}
    for text in texts:
        for word in analyzer(text):
            if word in word_counter:
                word_counter[word] += 1

    cnt = 0
    result_list = []
    for i, (word, _, df) in enumerate(word_value_list):
        result_list.append(word)
        cnt += word_counter[word]
        if cnt / total > rate:
            print(f'{i+1} words take {cnt / total} content.')
            break

    return result_list, analyzer


def main(rate, dataset='writing_prompts'):
    conds, texts = get_texts(dataset)

    vocab, analyzer = get_vocab(texts['train'], rate=rate)
    pickle.dump(vocab, open(f'data/{dataset}/vocab_{rate}.pickle', 'wb'))

    vocab_dict = {word: 1 for word in vocab}
    for split in ['train', 'dev', 'test']:
        print(f'extracting {split} set...')

        examples = []
        for cond, text in zip(conds[split], texts[split]):
            extracted_paras = []
            for para in text.split('\n'):
                extracted_paras.append(' '.join([
                    word for word in analyzer(para) if word in vocab_dict]))

            extracted_text = '\n'.join(extracted_paras)

            examples.append({
                'condition': cond,
                'extracted_text': extracted_text,
                'original_text': text
            })

        pickle.dump(examples, open(
            f'data/{dataset}/extracted_{split}_{rate}words.pickle', 'wb'))

        log_file = open(
            f'data/{dataset}/extracted_{split}_{rate}words.txt', 'w')

        for example in examples:
            print('CONDITION:{}\n\nEXTRACTED:\n{}\n\nORIGINAL TEXT:\n{}'.format(
                example['condition'],
                example['extracted_text'], example['original_text']),
                file=log_file)
            print('=' * 100, '\n\n', file=log_file)
            log_file.flush()


if __name__ == '__main__':
    fire.Fire(main)
