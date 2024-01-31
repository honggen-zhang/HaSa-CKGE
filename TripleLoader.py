#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:24:55 2023
@author: hz
"""
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import csv
import pandas as pd
import networkx as nx
import torch.utils.data.dataset
from typing import Optional, List
from transformers import AutoTokenizer
from tokens import get_tokenizer
from tqdm import tqdm


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    #tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               add_special_tokens=True,
                               max_length=30,
                               return_token_type_ids=True,
                               padding='max_length',
                               truncation=True)
    return encoded_inputs

class Entites:
    
    def __init__(self, entity):
        self.entity = entity

        
    def vectorize(self) -> dict:


        entity_encoded_inputs = _custom_tokenize(text=str(self.entity))
        #print('r_encoded_inputs',self.relation)

        return {'entity_token_ids': entity_encoded_inputs['input_ids'],
                'entity_token_type_ids': entity_encoded_inputs['token_type_ids'],
                'entity_token_mask': entity_encoded_inputs['attention_mask'],
                }
class Token:
    
    def __init__(self, head_id, relation, tail_id):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        
    def vectorize(self) -> dict:

        head_word = self.head_id 
        head_encoded_inputs = _custom_tokenize(text=str(head_word))

        tail_word =self.tail_id
        tail_encoded_inputs = _custom_tokenize(text=str(tail_word))
        
        r_encoded_inputs = _custom_tokenize(text=str(self.relation))

        return {'relation_token_ids': r_encoded_inputs['input_ids'],
                'relation_token_type_ids': r_encoded_inputs['token_type_ids'],
                'relation_token_mask': r_encoded_inputs['attention_mask'],
                'relation': str(self.relation),
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'tail_token_mask': tail_encoded_inputs['attention_mask'],
                'tail': str(tail_word),
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'head_token_mask': head_encoded_inputs['attention_mask'],
                'head': str(head_word),
                'obj': self}        

      
        
class TripleLoader(object):
    
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
    def get_examples(self,filename):
        examples = []

        df = pd.read_csv(os.path.join(self.dataset_folder, filename), usecols=(['head','relation','tail']))
        m,n = df.shape
        for i in tqdm(range(m)):
            row=df.iloc[i]

            s1 = row['head']
            s2 = row['relation']
            s3 = row['tail']
            examples.append(Token(s1, s2, s3))

        return examples
    
    
    def get_entities_label(self,filename):
        entities = []

            
        df = pd.read_csv(os.path.join(self.dataset_folder, filename), usecols=(['name']))
        m,n = df.shape
        for i in range(m):
            row=df.iloc[i]


            s3 = row['name']

            entities.append(s3)

        return entities    
    
    def get_entities(self,filename):
        entities = []

            
        df = pd.read_csv(os.path.join(self.dataset_folder, filename), usecols=(['name']))
        m,n = df.shape
        for i in tqdm(range(m)):
            row=df.iloc[i]

            #for id, row in enumerate(data):
            #print(row)
            s1 = ' '
            #row1=df.iloc[i-1]
            s2 = ' '
            #row2=df.iloc[i-2]
            s3 = row['name']

            entities.append(Entites(s3))


        return entities
    

class Graph_KG(object):
    
    def __init__(self, dataset_folder, has_header=False):
        self.dataset_folder = dataset_folder
        self.has_header = has_header
    
    def graph_generator(self, filename):
        G=nx.MultiDiGraph()
        df = pd.read_csv(os.path.join(self.dataset_folder, filename), usecols=(['head','relation','tail']))
        m,n = df.shape
        for i in tqdm(range(m)):
            row=df.iloc[i]

            #for id, row in enumerate(data):
            #print(row)
            s1 = row['head']
            s2 = row['relation']
            s3 = row['tail']
            G.add_edge(str(s1), str(s3))
    
        return G 

        
        
class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self,examples):
        #self.path = path
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()        
        
        
def collate_en(batch_data: List[dict]) -> dict:

    
    head_token_ids = to_indices_and_mask(
        [torch.LongTensor(ex['entity_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['entity_token_type_ids']) for ex in batch_data],
        need_mask=False)
    head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['entity_token_mask']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)

    #batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'entity_token_ids': head_token_ids,
        'entity_token_mask': head_mask,
        'entity_token_type_ids': head_token_type_ids,
        
    }

    return batch_dict       
      
        
def collate(batch_data: List[dict]) -> dict:
    r_token_ids = to_indices_and_mask(
        [torch.LongTensor(ex['relation_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    
    r_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['relation_token_type_ids']) for ex in batch_data],
        need_mask=False)
    r_mask = to_indices_and_mask(
        [torch.LongTensor(ex['relation_token_mask']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    

    tail_token_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)
    tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_mask']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    

    head_token_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)
    head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_mask']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)

    #batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'r_token_ids': r_token_ids,
        'r_mask': r_mask,
        'r_token_type_ids': r_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        
    }
    name_dict = {'head':[ex['head'] for ex in batch_data],
                 'relation':[ex['relation'] for ex in batch_data],
                 'tail':[ex['tail'] for ex in batch_data],}

    return batch_dict,name_dict

      
        
def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    for i, t in enumerate(batch_tensor):

        indices[i, :len(t)].copy_(t)

    return indices

        
        
        
        
        
        
        
        
        
        
        
