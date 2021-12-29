# -*- coding: utf - 8 -*-

import os
import json
import shutil
import collections

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import deepspeed

import utils
from models import get_model
from datasets import get_dataset


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.debug:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.configs.device
        else:
            os.environ['NCCL_SHM_DISABLE'] = '1'
            # deepspeed needs a dict config
            self.deepspeed_config = vars(configs.deepspeed)
        self.model_save_dir = os.path.join('_saved_models', self.configs.config)
        os.makedirs(self.model_save_dir, exist_ok=True)

    def run(self, mode):
        MAPPINGS = {
            'train': self.train,
            'eval': self.eval,
            'predict': self.predict
        }
        return MAPPINGS[mode].__call__()

    def train(self):
        configs = self.configs
        device = configs.local_rank

        dataset_cls = get_dataset(configs.task.dataset)
        train_dataset = dataset_cls(configs, 'train')
        train_sampler = DistributedSampler(
            dataset=train_dataset, num_replicas=torch.cuda.device_count(),
            rank=device, shuffle=True, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=configs.optim.train_batch_size,
            sampler=train_sampler, collate_fn=train_dataset.train_collate_fn
        )

        model_cls = get_model(configs.model.model_name)
        model = model_cls(configs)
        model.cuda(device)

        params = model.get_optimizer_params()
        if configs.debug:
            optimizer = AdamW(params, lr=configs.optim.lr, weight_decay=0.01)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=len(train_loader) * configs.optim.warmup_epochs,
                num_training_steps=len(train_loader) * configs.optim.epochs
            )
        else:
            model, _, _, _ = deepspeed.initialize(
                model=model, model_parameters=params, config=self.deepspeed_config
            )

        for epoch in range(1, configs.optim.epochs + 1):

            model.train()
            losses = []
            train_loader = tqdm(train_loader, ncols=0, desc='Train epoch %s/%s' % (epoch, configs.optim.epochs))
            for batch in train_loader:
                batch = utils.to_device(batch, device)
                loss = model(batch, task=configs.task.task_name)

                if configs.debug:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                else:
                    model.backward(loss)
                    model.step()

                losses.append(loss.item())
                train_loader.set_postfix({
                    'loss': str(round(sum(losses) / len(losses), 4))
                })

            self.save_model(model)

    def eval(self):
        configs = self.configs
        device = configs.local_rank

        dataset_cls = get_dataset(configs.task.dataset)
        eval_dataset = dataset_cls(configs, 'eval')
        eval_loader = DataLoader(
            eval_dataset, batch_size=configs.optim.eval_batch_size, shuffle=False,
            collate_fn=eval_dataset.predict_collate_fn
        )

        model_cls = get_model(configs.model.model_name)
        model = model_cls(configs)
        model = self.load_model(model, mode='eval')
        model.cuda(device)

        for batch in eval_loader:
            batch = utils.to_device(batch, device)
            model(batch, task='mlm_infer')

    def predict(self):
        configs = self.configs
        device = configs.local_rank

        dataset_cls = get_dataset(configs.task.dataset)
        predict_dataset = dataset_cls(configs, 'predict')
        predict_loader = DataLoader(
            predict_dataset, batch_size=configs.optim.predict_batch_size, shuffle=False,
            collate_fn=predict_dataset.predict_collate_fn
        )

        model_cls = get_model(configs.model)
        model = model_cls(configs)
        model.cuda(device)

        for batch in predict_loader:
            batch = utils.to_device(batch, device)
            model(batch)

    def load_model(self, model, mode):
        if mode == 'train':
            model_load_path = self.configs.task.model_load_path
        else:
            model_load_path = os.path.join(self.model_save_dir, 'model.states')
        model_states = torch.load(model_load_path, map_location="cpu")
        model.load_state_dict(model_states, strict=False)
        return model

    def save_model(self, model):
        save_path = os.path.join(self.model_save_dir, 'model.states')
        torch.save(model.state_dict(), save_path)
