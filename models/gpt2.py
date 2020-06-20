from collections import namedtuple
from tqdm import tqdm, trange
import random
import os

import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from .gpt2_utils import GPT2LMHeadModelMultiDevicesWrapper, sample_sequence


GPT2_MAX_LENGTH = 1024


TextDataExample = namedtuple('TextDataExample', ['cond', 'text'])


class GPT2:
    def __init__(self, gpt2_type):
        self._gpt2_type = gpt2_type
        self._tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)

        if gpt2_type in ['gpt2-medium', 'gpt2-large']:
            n_gpus = min(torch.cuda.device_count(),
                         2 if gpt2_type == 'gpt2-medium' else 4)

            devices = [f'cuda:{i}' for i in range(n_gpus)]
            self._model = GPT2LMHeadModelMultiDevicesWrapper(
                gpt2_type=gpt2_type, devices=devices)
        else:
            self._model = GPT2LMHeadModel.from_pretrained(gpt2_type).to('cuda')

        self._optimizer = None
        self._lr_scheduler = None
        self._global_step = 0

        self._dataset = {}
        self._eval_steps = None
        self._log_dir = None
        self._log_file = None
        self._best_dev_loss = None

    def creat_log_dir(self, eval_steps, label):
        self._log_dir = f'training_logs/{label}'
        self._eval_steps = eval_steps
        self._best_dev_loss = float('inf')

        os.makedirs(os.path.join(self._log_dir, 'ckpt_gens'), exist_ok=True)
        self._log_file = open(os.path.join(self._log_dir, 'log.txt'), 'w')

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
        print(f'Model saved in {path}.')

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path, map_location='cuda'))
        print(f'Model {path} loaded.')

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

    def load_data(self, split, conds, texts):
        self._dataset[split] = []
        for cond, text in zip(conds, texts):
            self._dataset[split].append(TextDataExample(cond=cond, text=text))

    def train_epoch(self, batch_size):
        assert 'train' in self._dataset
        self._model.train()

        random.shuffle(self._dataset['train'])
        for i in trange(0, len(self._dataset['train']), batch_size,
                        desc='Training Epoch'):
            batch = self._dataset['train'][i:i + batch_size]

            self._optimizer.zero_grad()
            for example in batch:
                tokens = self._tokenizer.encode(
                    example.cond + ' [SEP] ' + example.text + ' <|endoftext|>',
                    add_special_tokens=False, max_length=GPT2_MAX_LENGTH)

                inputs = torch.tensor([tokens]).to(device='cuda')

                loss = self._model(inputs, labels=inputs)[0] / batch_size
                loss.backward()

            self._optimizer.step()
            self._lr_scheduler.step()

            self._global_step += 1
            if self._global_step % self._eval_steps == 0:
                self.gen_log()

    def evaluate(self):
        assert 'dev' in self._dataset
        self._model.eval()

        loss_list = []
        for example in self._dataset['dev']:
            tokens = self._tokenizer.encode(
                example.cond + ' [SEP] ' + example.text + ' <|endoftext|>',
                add_special_tokens=False, max_length=GPT2_MAX_LENGTH)

            inputs = torch.tensor([tokens]).to(device='cuda')

            with torch.no_grad():
                loss = self._model(inputs, labels=inputs)[0]
            loss_list.append(loss.item())

        return sum(loss_list) / len(loss_list)

    def generate(self, cond, top_k, top_p):
        context_tokens = self._tokenizer.encode(
            cond + ' [SEP] ', add_special_tokens=False)

        out = sample_sequence(
            model=self._model,
            context=context_tokens,
            length=GPT2_MAX_LENGTH + 1 - len(context_tokens),
            top_k=top_k,
            top_p=top_p,
            device='cuda')

        out = out[:, len(context_tokens):].tolist()[0]
        text = self._tokenizer.decode(out, clean_up_tokenization_spaces=True)
        text = text[: text.find('<|endoftext|>')]

        return text

    def gen_log(self):
        eval_loss = self.evaluate()

        print(f'Global Step: {self._global_step}, Eval Loss: {eval_loss}',
              file=self._log_file)

        if eval_loss < self._best_dev_loss:
            self._best_dev_loss = eval_loss
            self.save_model(f'{self._log_dir}/best_model.pt')
            print('Best Model Updated.', file=self._log_file)

        self._log_file.flush()

        generation_file = open(
            f'{self._log_dir}/ckpt_gens/step{self._global_step}.txt', 'w')
        for example in self._dataset['dev'][:20]:
            truth_text = example.text
            gen_text = self.generate(cond=example.cond, top_k=-1, top_p=0.95)

            print(f'CONDITION:\n {example.cond}\n\n'
                  f'GENERATION:\n{gen_text}\n\n',
                  f'DATA:\n{truth_text}\n',
                  '=' * 100, '\n\n\n', file=generation_file)
            generation_file.flush()

    @property
    def train_dataset(self):
        return self._dataset['train']

    @property
    def get_lr(self):
        return self._lr_scheduler.get_lr()
