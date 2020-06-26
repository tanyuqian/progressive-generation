import os
from collections import deque
import pickle
from tqdm import trange

import torch
from torch.utils.data import Dataset

from transformers import GPT2Tokenizer


# ------------
# Data loading
# ------------


class CNNDataset(Dataset):
    """ Abstracts the dataset used to train seq2seq models.

    The class will process the documents that are located in the specified
    folder. The preprocessing will work on any document that is reasonably
    formatted. On the CNN/DailyMail dataset it will extract both the story
    and the summary.

    CNN/Daily News:

    The CNN/Daily News raw datasets are downloaded from [1]. The stories are
    stored in different files; the summary appears at the end of the story as
    sentences that are prefixed by the special `@highlight` line. To process
    the data, untar both datasets in the same folder, and pass the path to this
    folder as the "data_dir argument. The formatting code was inspired by [2].

    [1] https://cs.nyu.edu/~kcho/
    [2] https://github.com/abisee/cnn-dailymail/
    """

    def __init__(self, ):
        """ We initialize the class by listing all the documents to summarize.
        Files are not read in memory due to the size of some datasets (like CNN/DailyMail).
        """

        path = 'cnn/stories'
        if not os.path.exists(path):
            os.system('perl download/gdown.pl '
                      '\'https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ\' '
                      'cnn_stories.tgz')
            os.system('tar -xvf cnn_stories.tgz')

        assert os.path.isdir(path)

        self.documents = []
        story_filenames_list = sorted(os.listdir(path))
        for story_filename in story_filenames_list:
            if "summary" in story_filename:
                continue
            path_to_story = os.path.join(path, story_filename)
            if not os.path.isfile(path_to_story):
                continue
            self.documents.append(path_to_story)

    def __len__(self):
        """ Returns the number of documents. """
        return len(self.documents)

    def __getitem__(self, idx):
        document_path = self.documents[idx]
        document_name = document_path.split("/")[-1]
        with open(document_path, encoding="utf-8") as source:
            raw_story = source.read()
            story_lines, summary_lines = process_story(raw_story)
        return document_name, story_lines, summary_lines


def process_story(raw_story):
    """ Extract the story and summary from a story file.

    Attributes:
        raw_story (str): content of the story file as an utf-8 encoded string.

    Raises:
        IndexError: If the stoy is empty or contains no highlights.
    """
    nonempty_lines = list(filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split("\n")]))

    # for some unknown reason some lines miss a period, add it
    nonempty_lines = [_add_missing_period(line) for line in nonempty_lines]

    # gather article lines
    story_lines = []
    lines = deque(nonempty_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError:
            # if "@highlight" is absent from the file we pop
            # all elements until there is None, raising an exception.
            return story_lines, []

    # gather summary lines
    summary_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

    return story_lines, summary_lines


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', "\u2019", "\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."


def main():
    dataset = CNNDataset()

    output_dir = 'data/cnn'
    os.makedirs(output_dir)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

    split_size = {
        'train': 10000,
        'dev': 5000,
        'test': 1000
    }

    train_texts = []
    for i in trange(20000, desc='Getting Train Text'):
        _, story_lines, _ = dataset[i]
        text = '\n\n'.join(story_lines)
        if len(tokenizer.tokenize(text)) > 1022:
            continue
        train_texts.append({'condition': '', 'text': text})

        if len(train_texts) >= split_size['train']:
            break

    print('#train:', len(train_texts))
    pickle.dump(train_texts, open(
        os.path.join(output_dir, 'train.pickle'), 'wb'))

    dev_texts = []
    for i in trange(20000, 30000, desc='Getting Dev Text'):
        _, story_lines, _ = dataset[i]
        text = '\n\n'.join(story_lines)
        if len(tokenizer.tokenize(text)) > 1022:
            continue
        dev_texts.append({'condition': '', 'text': text})

        if len(dev_texts) >= split_size['dev']:
            break

    print('#dev:', len(dev_texts))
    pickle.dump(dev_texts, open(
        os.path.join(output_dir, 'dev.pickle'), 'wb'))

    test_texts = []
    for i in trange(30000, 40000, desc='Getting Test Text'):
        _, story_lines, _ = dataset[i]
        text = '\n\n'.join(story_lines)
        if len(tokenizer.tokenize(text)) > 1022:
            continue
        test_texts.append({'condition': '', 'text': text})

        if len(test_texts) >= split_size['test']:
            break

    print('#test:', len(test_texts))
    pickle.dump(test_texts, open(
        os.path.join(output_dir, 'test.pickle'), 'wb'))


if __name__ == '__main__':
    main()