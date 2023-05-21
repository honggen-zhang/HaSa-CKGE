#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:49:24 2023

@author: hz
"""

import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer,SentencesDataset
import torch
from torch import optim
from model import HaSa,HaSa_Hard_Bias
from loss import InBatch_hard_loss,Bert_batch_bias_hard,InBatch_hard_NCEloss
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from TripleLoader import TripleLoader,Dataset,collate,collate_en,Graph_KG
import csv
from negative_sample import NegativeSampleHard_check
from time import sleep
from tqdm import tqdm
import operator
import numpy as np
import util
import networkx as nx
import torch.nn.functional as F
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from random import sample
def dic_true_triple():
    all_triple = []
    triplet_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('train2name_sample.csv')
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

def dic_true_triple_rest():
    all_triple = []
    triplet_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('train2name_rest.csv')
    all_triple.extend(examples)
    #examples = triplet_reader.get_examples('valid2name.csv')
    #all_triple.extend(examples)
    #examples = triplet_reader.get_examples('test2name.csv')
    #all_triple.extend(examples)
    
    triple_dic_rest = {}
    for triple in all_triple:
        head = triple.head_id
        relation = triple.relation
        tail = triple.tail_id
        triple_dic_rest[head+'@'+relation] = []
    
    for triple in all_triple:
        head = triple.head_id
        relation = triple.relation
        tail = triple.tail_id
        tail_list = triple_dic_rest[head+'@'+relation]
        if tail not in tail_list:
            tail_list.append(tail)
        triple_dic_rest[head+'@'+relation] = tail_list
        
    return triple_dic_rest


def embedding_bank(entites_dataset_all,eneities_labels):
    energy.eval()
    dic_embedding_bank = {}
    dataloader_entity = DataLoader(entites_dataset_all, batch_size=1000,collate_fn=collate_en, shuffle=False, drop_last=False)
    entity_list = []
    for j, xx_e in enumerate(dataloader_entity): 
        

        xx_e = move_to_cuda(xx_e)
        entity_tensor = energy.module.predict_ent_embedding(**xx_e)
        entity_list.append(entity_tensor)
    entity_tensor_all = torch.cat(entity_list, dim=0)
    dic_embedding_bank = {eneities_labels[i]:entity_tensor_all[i] for i in range(len(eneities_labels))}
    
    return dic_embedding_bank
    
def true_entites(test_data_label,dic_triple):
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


def match_negatives(sample_entity_dict):
    false_negative_tails = {}
    for hr, t_list in sample_entity_dict.items():
        try:
            true_tails = dic_triplet_rest[hr]
            set_inter = set(true_tails).intersection(set(t_list))
            false_negative_tails[hr] =  list(set_inter)
        except:
            false_negative_tails[hr] = []
            

    return false_negative_tails
    


    
    

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

batch_size = 128
num_hard_neg = batch_size*2
num_false_neg = 4
data_name = 'FB15K237_L/'
data_name = 'WN18RR/' 
torch.cuda.empty_cache()
device = torch.device('cuda')
pretrain_model = 'sentence-transformers/all-mpnet-base-v2'
input_dim = 768
em_dim = 500
energy = HaSa(pretrain_model,input_dim,em_dim).to(device)


print('Resuming from checkpoint at ckpts/nce.pth.tar...')
energy = torch.nn.DataParallel(energy).cuda()

triplet_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name+'BERT_a/')
examples = triplet_reader.get_examples('train2name_sample.csv')
train_dataset = Dataset(examples = examples)
dataloader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate, pin_memory=True, shuffle=True, drop_last=True)
G= Graph_KG( 'hasa/data/benchmarks/'+data_name+'BERT_a/').graph_generator('train2name_sample.csv')  
dic_triplet_rest_ = dic_true_triple_rest()
dic_triplet = dic_true_triple()


entity_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name)
eneities = entity_reader.get_entities('entity2name.csv')
eneities_label = entity_reader.get_entities_label('entity2name.csv')
print('making graph')
graph_stru = util.ego_graph_dic_all(eneities_label, G,length = 3)
print('graph down---------------------------')
entity_dataset = Dataset(examples = eneities)

os_file = 'dynamicgraph_lts/Honggen/HaSa/tmp/model_sample/WN/'
filelist_folder = os.listdir(os_file)
filelist_folder.sort() 
print(filelist_folder[-1:])
for model_file_name in filelist_folder[-3:-2]:

    checkpoint = torch.load(os_file+model_file_name)
    energy.load_state_dict(checkpoint['energy'])


    embedding_bank_ = embedding_bank(entity_dataset,eneities_label)
    negative_sample = NegativeSampleHard_check(batch_size,num_false_neg,num_hard_neg,entity_dataset,eneities_label,graph_stru,dic_triplet)

    #NegativeSampleRandom_check(batch_size,num_false_neg,entity_dataset,eneities_label,dic_triplet)
    data_negatives = []
    for epoch in range(19,20):

        for i, xy in enumerate(tqdm(dataloader)): 
            x = xy[0]
            y = xy[1]
            tail_true_id = util.true_entites(y,dic_triplet,eneities_label)
            #dict_data_tail,_=negative_sample.neg_sampling_check(energy,xy,embedding_bank_,
                                                                       tail_true_id,dic_triplet_rest_)
            _,dict_data_tail=negative_sample.neg_sampling_check(energy,xy,embedding_bank_,
                                                                       tail_true_id,dic_triplet_rest_)
            data_negatives.append(dict_data_tail)
            if i>300:
                break
    print(len(data_negatives))
    
    
    true_neg_keys = []
    for dict_data in data_negatives:
        dict_false_tail_label = dict_data['true_tail_lable']
        subnodes = list(dict_false_tail_label.values())[0]
        #G0 = G.subgraph(subnodes)
        for key, value_list in dict_false_tail_label.items():

            head = key.split('@')[0]
            if len(value_list)>0:
                for tail in value_list:
                    true_neg_keys.append(tail)
    
    false_neg_keys = []
    for dict_data in data_negatives:
        dict_false_tail_label = dict_data['false_tail_lable']
        subnodes = list(dict_false_tail_label.values())[0]
        #G0 = G.subgraph(subnodes)
        for key, value_list in dict_false_tail_label.items():

            head = key.split('@')[0]
            if len(value_list)>0:
                for tail in value_list:
                    false_neg_keys.append(tail)
    
    true_neg_keys_sample = sample(true_neg_keys,len(false_neg_keys))
    
    


    score_true_all = []
    for dict_data in data_negatives:
        dict_false_tail_score = dict_data['true_tail_score']
        for key, value_list in dict_false_tail_score.items():
            score_true_all.extend(value_list)

    score_false_all = []
    for dict_data in data_negatives:
        dict_false_tail_score = dict_data['false_tail_score']
        for key, value_list in dict_false_tail_score.items():
            score_false_all.extend(value_list)


    true_length = []
    for dict_data in data_negatives:
        dict_false_tail_label = dict_data['true_tail_lable']
        subnodes = list(dict_false_tail_label.values())[0]
        #G0 = G.subgraph(subnodes)
        for key, value_list in dict_false_tail_label.items():

            head = key.split('@')[0]
            if len(value_list)>0:
                for tail in value_list:
                    #if tail in true_neg_keys_sample:
                    if 1==1:
                    #print(head)
                    #print(tail)
                    #print(nx.shortest_path_length(G, source=head, target=tail))
                        try:
                            len_p = 25
                            #len_p = nx.shortest_path_length(G, source=head, target=tail)
                        except:
                            len_p = 25
                        true_length.append(len_p)

    false_length = []
    
    for dict_data in data_negatives:
        dict_false_tail_label = dict_data['false_tail_lable']
        for key, value_list in dict_false_tail_label.items():
            subnodes = list(dict_false_tail_label.values())[0]
            #G0 = G.subgraph(subnodes)
            head = key.split('@')[0]
            if len(value_list)>0:
                for tail in value_list:
                    try:
                        len_p = 25
                        #len_p = nx.shortest_path_length(G, source=head, target=tail)
                    except:
                        len_p = 25
                    false_length.append(len_p)
    import seaborn as sns

    label_true = ['true_negative']*len(score_true_all)
    label_false = ['false_negative']*len(score_false_all)
    print('num_false:',len(label_false))
    print('ave_num_false:',len(label_false)/len(data_negatives))
    print('ratio:',len(label_false)/len(label_true))
    d = {'Ds': sample(score_true_all,len(score_false_all))+score_false_all, 'Dt': sample(true_length,len(score_false_all))+false_length,'label':label_true[:len(score_false_all)]+label_false}

    #d = {'Ds': score_true_all+score_false_all, 'Dt': true_length+false_length,'label':label_true+label_false}

    df = pd.DataFrame(data=d)
    plt.figure()
    sns.histplot(data=df, x="Ds", hue="label",bins=200,stat="density",kde=True)
    #sns.displot(data=df, x="Dt", hue="label", fill=True, kind="kde")
    #sns.jointplot(data=df, x="Ds", y="Dt", hue="label",alpha = 0.8)
    #plt.xlabel('$d(h,t, A)$',fontsize=25)
    plt.xlabel('$\log\phi(h,r,t)$',fontsize=25)
    #plt.xlim(0, 6)
    plt.tight_layout()
    plt.ylabel('Density',fontsize=25)
    plt.savefig('dynamicgraph_lts/Honggen/HaSa/tmp/model_sample/WN/wn_stru_'+model_file_name+'.png',dpi = 200)
    #plt.scatter(score_true_all,true_length,s = 3);plt.scatter(score_false_all,false_length,s =3)
