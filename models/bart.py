from collections import namedtuple
import random
from tqdm import tqdm, trange
import os
import nltk

import torch

from fairseq.sequence_generator import SequenceGenerator

from transformers import AdamW, get_linear_schedule_with_warmup

from .bart_utils import BARTModelWrapper


BART_MAX_LEN = 1024


TextPairData = namedtuple('TextPairData', ['src_text', 'tgt_text'])


class BART:
    def __init__(self):
        self._model = BARTModelWrapper()

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}

        self._log_dir = None
        self._eval_steps = None
        self._log_file = None
        self._best_dev_loss = None

    def create_training_log(self, eval_steps, label):
        self._log_dir = f'{label}_training_logs'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self._log_dir, 'ckpt_gens'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

    def get_optimizer(self, lr, train_steps, warmup_steps,
                      weight_decay, adam_epsilon):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        self._optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        self._lr_scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=train_steps)

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._model.load_state_dict(
            torch.load(path, map_location='cuda'))
        print(f'Model {path} loaded.')

    def load_data(self, set_type, src_texts, tgt_texts):
        assert len(src_texts) == len(tgt_texts)

        self._dataset[set_type] = []
        for src_text, tgt_text in tqdm(zip(src_texts, tgt_texts),
                                       total=len(src_texts),
                                       desc=f'loading {set_type} data'):
            self._dataset[set_type].append(TextPairData(
                src_text=src_text, tgt_text=tgt_text))

    def train_epoch(self, batch_size, noise_vocab):
        assert 'train' in self._dataset

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='BART Training'):
            self._model.split_to_gpus(2)
            self._model.train()

            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()

            for example in batch:
                noised_src_text = self._add_noise(example.src_text, noise_vocab)

                loss = self._get_seq2seq_loss(
                    src_text=noised_src_text, tgt_text=example.tgt_text)
                loss = loss / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            self._global_step += 1
            if self._global_step % self._eval_steps == 0:
                self.gen_log()

    def evaluate(self):
        assert 'dev' in self._dataset
        self._model.split_to_gpus(1)
        self._model.eval()

        loss_list = []
        for example in self._dataset['dev']:
            with torch.no_grad():
                loss = self._get_seq2seq_loss(
                    src_text=example.src_text, tgt_text=example.tgt_text)

            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, src_text, top_k, top_p):
        self._model.set_mode('infer')
        self._model.eval()

        generator = SequenceGenerator(
            tgt_dict=self._model.dictionary,
            max_len_b=BART_MAX_LEN,
            sampling=True,
            sampling_topk=top_k,
            sampling_topp=top_p)

        src_tokens = self._model.encode(src_text)[:BART_MAX_LEN]

        outputs = generator.generate(
            models=[self._model.model],
            sample={'net_input': {
                'src_tokens': src_tokens.unsqueeze(0).to('cuda'),
                'src_lengths': torch.tensor([len(src_tokens)]).to('cuda')
            }})

        return self._model.decode(outputs[0][0]['tokens'].cpu())

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/models/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        self._log_file.flush()

        generation_file = open(
            f'{self._log_dir}/generations/step{self._global_step}.txt', 'w')

        # for src_text, tgt_text in zip(src_texts[:5], tgt_texts[:5]):
        for example in self._dataset['dev'][:10]:
            gen_text = self.generate(example.src_text, top_k=-1., top_p=0.95)

            print('SOURCE:\n', example.src_text, '\n', '-' * 50, '\n',
                  'GENERATION:\n', gen_text, '\n', '-' * 50, '\n',
                  'TARGET:\n', example.tgt_text, '\n', '=' * 100, '\n\n\n',
                  file=generation_file)
            generation_file.flush()

    def _get_seq2seq_loss(self, src_text, tgt_text):
        src_tokens = self._model.encode(src_text)[:BART_MAX_LEN]
        tgt_tokens = self._model.encode(tgt_text)[:BART_MAX_LEN]

        logits, extra = self._model(
            src_tokens=src_tokens.unsqueeze(0),
            src_lengths=[src_tokens.shape[0]],
            prev_output_tokens=tgt_tokens.unsqueeze(0))

        tgt_tokens = tgt_tokens.to(logits.device)

        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = tgt_tokens[:, 1:].contiguous()

        # Flatten the tokens
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self._model.dictionary.pad())
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def _add_noise(self, text, noise_vocab):
        if text.find(' [SEP] ') == -1:
            return text
        prompt = text.split(' [SEP] ')[0]
        template = text.split(' [SEP] ')[1]

        noised_paras = []
        for para in template.split('\n'):
            words = nltk.word_tokenize(para)
            for i in range(len(words)):
                if random.random() < 0.1:
                    words[i] = random.choice(noise_vocab)
                    if i + 1 < len(words) and random.random() < 0.5:
                        words[i + 1] = random.choice(noise_vocab)
                        if i + 2 < len(words) and random.random() < 0.5:
                            words[i + 2] = random.choice(noise_vocab)
                            if i + 3 < len(words) and random.random() < 0.5:
                                words[i + 3] = random.choice(noise_vocab)

            noised_paras.append(' '.join(words))

        result = prompt + ' [SEP] ' + '\n'.join(noised_paras)

        return result

    @property
    def dataset(self):
        return self._dataset

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()
