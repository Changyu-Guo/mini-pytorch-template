# -*- coding: utf - 8 -*-

import os

from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from models import get_model_cls
from datasets import get_dataset_cls
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils


class DDPTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device('cuda', int(os.getenv('LOCAL_RANK')))
        os.environ['NCCL_SHM_DISABLE'] = '1'

    def run(self, mode):
        MAPPINGS = {
            'train': self.train,
            'eval': self.eval,
            'predict': self.predict
        }
        return MAPPINGS[mode]()

    def train(self):
        configs = self.configs

        # ===== 1. init process group =====
        dist.init_process_group(
            backend='nccl', world_size=int(os.getenv('WORLD_SIZE')), rank=int(os.getenv('RANK')),
            init_method='tcp://' + os.getenv('MASTER_ADDR') + ':' + os.getenv('MASTER_PORT')
        )

        # ===== 2. load data =====
        dataset_cls = get_dataset_cls(configs.setup.dataset_cls)
        train_dataset = dataset_cls(configs, 'train')

        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=configs.trainer_configs.train_batch_size_per_gpu,
            drop_last=True, sampler=train_sampler,
            collate_fn=train_dataset.train_collate_fn,
        )

        # ===== 3. load model and optimizer =====
        model, optimizer, scheduler = self.setup_model_and_optimizer(len(train_loader))
        model = DistributedDataParallel(
            module=model,
            device_ids=[int(os.getenv('LOCAL_RANK'))],
            output_device=int(os.getenv('LOCAL_RANK'))
        )

        # ===== 4. train model =====
        rank = int(os.getenv('RANK'))
        for epoch in range(1, configs.trainer_configs.epochs + 1):
            train_sampler.set_epoch(epoch)

            model.train()
            losses = []
            train_loader = tqdm(
                train_loader, ncols=0, disable=rank != 0,
                desc='Train epoch %s/%s' % (epoch, configs.trainer_configs.epochs)
            )

            for batch in train_loader:
                batch = utils.to_device(batch, self.device)
                optimizer.zero_grad()
                loss = model(batch, task=configs.task_name)
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                train_loader.set_postfix({
                    'loss': str(round(sum(losses) / len(losses), 4))
                })

            if rank == 0:
                model.module.save_checkpoint()

    def eval(self):
        pass

    def predict(self):
        pass

    def setup_model_and_optimizer(self, num_batches):
        model = self.get_model()
        optimizer = self.get_optimizer(model.get_optimizer_params())
        lr_scheduler = self.get_lr_scheduler(optimizer, num_batches)
        return model, optimizer, lr_scheduler

    def get_model(self):
        model_cls = get_model_cls(self.configs.setup.model_cls)
        model = model_cls(self.configs)
        model.load_checkpoint()
        model.to(self.device)
        return model

    def get_optimizer(self, params):
        trainer_configs = self.configs.trainer_configs.optimizer_configs
        optimizer = get_optimizer(
            name=self.configs.trainer_configs.optimizer,
            params=params, configs=vars(trainer_configs)
        )
        return optimizer

    def get_lr_scheduler(self, optimizer, num_batches):
        num_warmup_steps = num_batches * self.configs.trainer_configs.warmup_epochs
        num_training_steps = num_batches * self.configs.trainer_configs.epochs

        scheduler_configs = vars(self.configs.trainer_configs.scheduler_configs)
        scheduler_configs.update({
            'num_warmup_steps': num_warmup_steps,
            'num_training_steps': num_training_steps
        })

        lr_scheduler = get_scheduler(
            name=self.configs.trainer_configs.scheduler,
            optimizer=optimizer, configs=scheduler_configs
        )

        return lr_scheduler
