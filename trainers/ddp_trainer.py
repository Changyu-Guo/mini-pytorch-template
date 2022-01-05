# -*- coding: utf - 8 -*-

import os
import json

from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from models import get_model_cls
from datasets import get_dataset_cls
import utils


class DDPTrainer:
    def __init__(self, configs):
        self.configs = configs
        os.environ['NCCL_SHM_DISABLE'] = '1'
        self.model_save_dir = os.path.join('_saved_models', self.configs.config)
        os.makedirs(self.model_save_dir, exist_ok=True)

    def run(self, mode):
        MAPPINGS = {
            'train': self.train,
            'eval': self.eval,
            'predict': self.predict
        }
        return MAPPINGS[mode]()

    def train(self):
        configs = self.configs
        device = torch.device('cuda', configs.local_rank)

        dist.get_world_size()

        dataset_cls = get_dataset_cls(configs.task.dataset_cls)
        train_dataset = dataset_cls(configs, 'train')

        dist.init_process_group(
            backend='nccl', world_size=int(os.getenv('WORLD_SIZE')), rank=int(os.getenv('RANK')),
            init_method='tcp://' + os.getenv('MASTER_ADDR') + ':' + os.getenv('MASTER_PORT')
        )

        train_sampler = DistributedSampler(
            dataset=train_dataset, num_replicas=int(os.getenv('WORLD_SIZE')),
            rank=int(os.getenv('RANK')), shuffle=True, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=configs.optim.train_batch_size_per_gpu,
            pin_memory=True, sampler=train_sampler, drop_last=True,
            collate_fn=train_dataset.train_collate_fn,
            num_workers=8
        )

        model_cls = get_model_cls(configs.model.model_cls)
        model = model_cls(configs)
        model = self.load_model(model, 'train')
        model.to(device)

        params = model.get_optimizer_params()
        step_samples = configs.optim.train_batch_size_per_gpu * configs.optim.gradient_accumulation_steps * int(os.getenv('WORLD_SIZE'))
        steps_per_epoch = len(train_dataset) // step_samples
        optimizer = AdamW(params, lr=configs.optim.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=steps_per_epoch * configs.optim.warmup_epochs,
            num_training_steps=steps_per_epoch * configs.optim.epochs
        )

        model = DistributedDataParallel(
            module=model,
            device_ids=[configs.local_rank],
            output_device=configs.local_rank
        )

        rank = int(os.getenv('RANK'))
        for epoch in range(1, configs.optim.epochs + 1):
            train_sampler.set_epoch(epoch)

            model.train()
            losses = []

            if rank == 0:
                train_loader = tqdm(
                    train_loader, ncols=0,
                    desc='Train epoch %s/%s' % (epoch, configs.optim.epochs)
                )

            for batch in train_loader:
                batch = utils.to_device(batch, device)
                optimizer.zero_grad()
                loss = model(batch, task=configs.task.task_name)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if rank == 0:
                    losses.append(loss.item())
                    train_loader.set_postfix({
                        'loss': str(round(sum(losses) / len(losses), 4))
                    })

            if rank == 0:
                self.save_model(model.module)

    def eval(self):
        pass

    def predict(self):
        pass

    def load_model(self, model, mode):
        if mode == 'train':
            model_load_path = self.configs.task.model_load_path
        else:
            model_load_path = os.path.join(self.model_save_dir, 'model.states')

        model_states = torch.load(model_load_path, map_location='cpu')
        model.load_state_dict(model_states, strict=False)
        return model

    def save_model(self, model):
        save_path = os.path.join(self.model_save_dir, 'model.states')
        torch.save(model.state_dict(), save_path)
