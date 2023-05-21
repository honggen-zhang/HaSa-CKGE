#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:22:25 2023

@author: hz
"""
from transformers import AutoTokenizer

tokenizer: AutoTokenizer = None

def build_tokenizer():
    global tokenizer
    pretrain_model = 'sentence-transformers/all-mpnet-base-v2'
    #pretrain_model = 'sentence-transformers/bert-base-nli-mean-tokens'
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        #logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer()
    return tokenizer  
