# -*- coding: utf - 8 -*-

import os
import re
import json
import random
import collections

import torch
import jieba
from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BertTokenizer

from utils import parallel_apply


class MLMDataset:

    def __init__(self, configs, dataset_label):
        self.configs = configs
        self.dataset_label = dataset_label
        assert dataset_label in ['train', 'eval', 'predict']
        self.en_dataset_file = 'data/parallel-corpus/translation2019/en.txt'
        self.zh_dataset_file = 'data/parallel-corpus/translation2019/zh.txt'
        self.tokenizer = BertTokenizer(
            vocab_file=configs.vocab_file
        )
        self.mlm_probability = 0.15

        MAPPINGS = {
            'train': self.read_train,
            'eval': self.read_eval,
            'predict': self.read_predict
        }
        MAPPINGS[dataset_label]()

    def read_train(self):
        def yield_examples(batch_count):
            count, en_texts, zh_texts = 0, [], []
            en_reader = open(self.en_dataset_file, encoding='utf8')
            zh_reader = open(self.zh_dataset_file, encoding='utf8')
            for en_line, zh_line in zip(en_reader, zh_reader):
                en_texts.append(en_line.strip())
                zh_texts.append(zh_line.strip())
                count += 1
                if count == batch_count:
                    yield en_texts, zh_texts
                    count, en_texts, zh_texts = 0, [], []
            if en_texts:
                yield en_texts, zh_texts

        def make_features(texts):
            max_length = self.configs.task.seq_len
            features = []
            for en_text, zh_text in zip(*texts):
                en_input_ids = self.tokenizer.encode(en_text)[0]
                zh_input_ids = self.tokenizer.encode(zh_text)[0]
                length = len(en_input_ids) + len(zh_input_ids)
                if length > max_length:
                    continue
                en_position_ids = list(range(len(en_input_ids)))
                zh_position_ids = list(range(len(zh_input_ids)))
                input_ids = en_input_ids + zh_input_ids
                position_ids = en_position_ids + zh_position_ids
                mask_signals = [0] * (len(en_input_ids) + 1) + [1] * (len(zh_input_ids) - 2) + [0]
                features.append({
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'mask_signals': mask_signals
                })
            return features

        if self.configs.debug:
            feature_batches = []
            generator = yield_examples(20)
            examples = next(generator)
            feature_batches.append(make_features(examples))
        else:
            feature_batches = parallel_apply(
                make_features,
                yield_examples(10000),
                num_workers=20,
                max_queue_size=2000,
            )

        features = []
        for feature_batch in feature_batches:
            for feature in feature_batch:
                features.append(feature)

        self._features = features

    def read_eval(self):
        return self.read_train()

    def read_predict(self):
        return self.read_train()

    def get_train_item(self, index):
        return self.features[index]

    def get_eval_item(self, index):
        return self.features[index]

    def get_predict_item(self, index):
        return self.features[index]

    def train_collate_fn(self, features):
        # 1. PAD
        # 2. MASK
        batch = collections.defaultdict(list)
        keys = features[0].keys()
        for feature in features:
            for key in keys:
                batch[key].append(feature[key])

        self.pad_sequences(batch, pad_value=self.tokenizer.token_to_id('[PAD]'))

        self.mask_tokens(batch)

        return {
            'text_input_ids': torch.as_tensor(batch['input_ids'], dtype=torch.long),
            'text_position_ids': torch.as_tensor(batch['position_ids'], dtype=torch.long),
            'text_attention_mask': torch.as_tensor(batch['attention_mask'], dtype=torch.long),
            'text_labels': torch.as_tensor(batch['labels'], dtype=torch.long)
        }

    def eval_collate_fn(self, features):
        return self.train_collate_fn(features)

    def predict_collate_fn(self, features):
        return self.train_collate_fn(features)

    def __getitem__(self, index):
        if self.dataset_label == 'train':
            return self.get_train_item(index)
        elif self.dataset_label == 'eval':
            return self.get_eval_item(index)
        elif self.dataset_label == 'predict':
            return self.get_predict_item(index)

    @property
    def features(self):
        return self.__getattribute__('_features')

    def __len__(self):
        return len(self.features)

    def pad_sequences(self, batch, pad_value=0):
        batch['attention_mask'] = []
        max_length = max(len(x) for x in batch['input_ids'])
        for i in range(len(batch['input_ids'])):
            length = len(batch['input_ids'][i])
            diff = max_length - length
            batch['input_ids'][i] += [pad_value] * diff
            batch['position_ids'][i] += [0] * diff
            batch['mask_signals'][i] += [0] * diff
            batch['attention_mask'].append([1] * length + [0] * diff)

    def mask_tokens(self, batch):
        input_ids = torch.tensor(batch['input_ids'], dtype=torch.long)
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        mask_signals = torch.tensor(batch['mask_signals'], dtype=torch.bool)

        probability_matrix.masked_fill_(~mask_signals, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.token_to_id('[MASK]')

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.token_to_id('a'), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        batch['input_ids'] = input_ids
        batch['labels'] = labels
