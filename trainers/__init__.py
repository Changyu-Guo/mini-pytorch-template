# -*- coding: utf - 8 -*-

from trainers.gpu_trainer import GPUTrainer
from trainers.ddp_trainer import DDPTrainer

TRAINER_MAPPINGS = {
    'gpu_trainer': GPUTrainer,
    'ddp_trainer': DDPTrainer
}


def get_trainer_cls(name):
    return TRAINER_MAPPINGS[name]
