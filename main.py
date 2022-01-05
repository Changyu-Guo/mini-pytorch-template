# -*- coding: utf - 8 -*-

import argparse

from args import get_parser
from trainers import get_trainer_cls
import utils


def main(mode):
    parser = get_parser(mode)
    args, _ = parser.parse_known_args()
    configs = utils.get_configs(args)
    trainer_cls = get_trainer_cls(configs.task.trainer)
    trainer = trainer_cls(configs)
    trainer.run(mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', type=str, choices=['train', 'eval', 'predict'],
        help='Mode: `train`, `eval` or `predict`'
    )
    args, _ = parser.parse_known_args()
    main(args.mode)
