# -*- coding: utf - 8 -*-

import os
import json
import random
from types import SimpleNamespace

import torch
import numpy as np


def get_configs(args):
    file = os.path.join('configs', '%s.json' % args.config)
    configs = json.load(open(file), object_hook=lambda x: SimpleNamespace(**x))
    for key, value in vars(args).items():
        setattr(configs, key, value)
    return configs


def seed(seed=random.randint(0, 99999)):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_device(batch, device=0):
    converted_batch = dict()
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            converted_batch[key] = batch[key].to(device)
        else:
            converted_batch[key] = batch[key]

    return converted_batch


def parallel_apply_generator(f, iterable, num_workers, max_queue_size):
    from multiprocessing import Pool, Queue
    from queue import Full

    input_queue, output_queue = Queue(max_queue_size), Queue()

    # 若干生产者进程
    def worker_step(input_queue, output_queue):
        while True:
            index, data = input_queue.get()
            res = f(data)
            output_queue.put((index, res))

    pool = Pool(num_workers, worker_step, (input_queue, output_queue))

    input_count, output_count = 0, 0
    for index, data in enumerate(iterable):
        input_count += 1
        while True:
            try:
                input_queue.put((index, data), block=False)
                break
            except Full:
                while output_queue.qsize() > max_queue_size:
                    yield output_queue.get()
                    output_count += 1

        if output_queue.qsize() > 0:
            yield output_queue.get()
            output_count += 1

    while output_count != input_count:
        yield output_queue.get()
        output_count += 1

    pool.terminate()


def parallel_apply(f, iterable, num_workers, max_queue_size):
    generator = parallel_apply_generator(
        f, iterable, num_workers, max_queue_size
    )
    return [d for i, d in generator]
