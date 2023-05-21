#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:42:48 2022

@author: user1
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from tqdm import tqdm
import operator
from TripleLoader import TripleLoader,Dataset,collate,collate_en


def ego_graph_dic_all_yg(entities, G, length = 4):
    ego_dic = {}
    #nodes_rank = nx.degree_centrality(G)
    #nodes_rank_sorted = sorted(nodes_rank.items(),key = operator.itemgetter(1),reverse = True)
    #entities = nodes_rank_sorted.keys()
    c = 0
    for entity in tqdm(entities):
        nodes_ego = {}
        for k in range(2,length):#1 2 3
            #H = nx.ego_graph(G, n = entity, radius = 3)
            
            #nodes_ego[str(k)] = list(H.nodes)
            #print(entity)
            try:
                edges = list(nx.bfs_edges(G, source=entity, reverse = False, depth_limit=k))
           
                depth_node = []
                for s, t in edges:
                    depth_node.append(t)
                    if len(depth_node)>5:
                        break
            except:
                edges = entity[k+c:k+c+5]
                c = c+5
                depth_node=edges
            #edges = list(nx.bfs_edges(G, source=entity, reverse = True, depth_limit=k))
            #for s, t in edges:
                #depth_node.append(t)
                
            nodes_ego[str(k)] = list(set(depth_node))
        #nodes_ego[str(length-1)] = entities
        #nodes_ego[str(length)] = entities
        #nodes_ego[str(4)] = entities
        #print(len(H.nodes))
        ego_dic[entity] = nodes_ego
    #ego_dic_df = pd.DataFrame.from_dict(ego_dic, orient='index')
    #ego_dic_df.to_csv('/home/user1/Desktop/NCE/KG_NCE/ini/ego_dic.csv')
    return ego_dic

def ego_graph_dic_all(entities, G, length = 4):
    ego_dic = {}
    #nodes_rank = nx.degree_centrality(G)
    #nodes_rank_sorted = sorted(nodes_rank.items(),key = operator.itemgetter(1),reverse = True)
    #entities = nodes_rank_sorted.keys()
    c = 0
    for entity in tqdm(entities):
        nodes_ego = {}
        for k in range(1,length):#1 2 3
            #H = nx.ego_graph(G, n = entity, radius = 3)
            
            #nodes_ego[str(k)] = list(H.nodes)
            #print(entity)
            try:
                edges = list(nx.bfs_edges(G, source=entity, reverse = False, depth_limit=k))
           
                depth_node = []
                for s, t in edges:
                    depth_node.append(t)
            except:
                edges = entity[k+c:k+c+5]
                c = c+5
                depth_node = []
                for t in edges:
                    depth_node.append(t)
            #edges = list(nx.bfs_edges(G, source=entity, reverse = True, depth_limit=k))
            #for s, t in edges:
                #depth_node.append(t)
                
            nodes_ego[str(k)] = list(set(depth_node))
        #nodes_ego[str(length-1)] = entities
        #nodes_ego[str(length)] = entities
        #nodes_ego[str(4)] = entities
        #print(len(H.nodes))
        ego_dic[entity] = nodes_ego
    #ego_dic_df = pd.DataFrame.from_dict(ego_dic, orient='index')
    #ego_dic_df.to_csv('/home/user1/Desktop/NCE/KG_NCE/ini/ego_dic.csv')
    return ego_dic


      

def move_to_cuda(batchsample):
    if len(batchsample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(batchsample)  

def dic_true_triple(data_name,train_name):
    all_triple = []
    triplet_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name+'BERT_a/')
    examples = triplet_reader.get_examples(train_name)
    all_triple.extend(examples)

    
    triple_dic = {}
    for triple in all_triple:
        head = triple.head_id
        relation = triple.relation
        tail = triple.tail_id
        triple_dic[head+'@'+relation] = []
    
    for triple in all_triple:
        head = triple.head_id
        relation = triple.relation
        tail = triple.tail_id
        tail_list = triple_dic[head+'@'+relation]
        if tail not in tail_list:
            tail_list.append(tail)
        triple_dic[head+'@'+relation] = tail_list
        
    return triple_dic

def true_entites(test_data_label,dic_triple,eneities_label):
    head_e = test_data_label['head']
    relation = test_data_label['relation']
    tail_e = test_data_label['tail']
    length = len(head_e)
    all_true_tail = []
    for i in range(length):
        h = head_e[i]
        r = relation[i]
        t = tail_e[i]
        true_t = dic_triple[h+'@'+r]
        inx_i_list = []
        for ent in true_t:
            k = eneities_label.index(ent)
            inx_i_list.append(k)
            
        
        #true_t_ = triplet_reader.get_entities_e(true_t)
        all_true_tail.append(inx_i_list)
    return all_true_tail