# -*- coding: utf - 8 -*-

import argparse


def _add_common_args(parser):
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    return parser


def train_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    return parser


def eval_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    return parser


def predict_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    return parser


PARSER_MAPPINGS = {
    'train': train_parser,
    'eval': eval_parser,
    'predict': predict_parser
}


def get_parser(mode):
    return PARSER_MAPPINGS[mode]()
