# -*- coding: utf - 8 -*-

import torch
from torch import nn

from transformers.models.bert import BertConfig
from transformers.models.bert import BertForMaskedLM


class BertModel(nn.Module):
    def __init__(self, configs):
        super(BertModel, self).__init__()

        bert_config = BertConfig.from_json_file(configs.bert_config_file)
        self.bert = BertForMaskedLM(bert_config)

    def forward(self, task='train'):
        TASKS = {
            'train': self.task_train,
            'eval': self.task_eval
        }
        return TASKS[task]()

    def task_train(self):
        pass

    def task_eval(self):
        pass
