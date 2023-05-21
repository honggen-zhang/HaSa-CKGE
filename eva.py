#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:25:13 2023

@author: hz
"""

import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer,SentencesDataset
import torch
from model import HaSa,HaSa_Hard_Bias
from loss import InBatch_hard_loss,Bert_batch_bias_hard,InBatch_hard_NCEloss,Bert_batch_bias_wo_hard
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from TripleLoader import TripleLoader,Dataset,collate,collate_en,Graph_KG
import csv
from tqdm import tqdm
import operator
import numpy as np
import util
import torch.nn.functional as F
import pandas as pd



def dic_true_triple(data_name):
    all_triple = []
    triplet_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('test2name.csv')
    all_triple.extend(examples)
    examples = triplet_reader.get_examples('train2name.csv')
    all_triple.extend(examples)
    examples = triplet_reader.get_examples('valid2name.csv')
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

def true_entites(test_data_label,dic_triple,eneities_labels):
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
        inx_i_list = [eneities_labels.index(h)]
        for ent in true_t:
            k = eneities_labels.index(ent)
            inx_i_list.append(k)
            
        
        #true_t_ = triplet_reader.get_entities_e(true_t)
        all_true_tail.append(inx_i_list)
    return all_true_tail

def ture_scores(true_tail_list,hr_vector):
    all_p_score = []
    for i in range(len(true_tail_list)):
        dataloader_en = DataLoader(Dataset(examples = true_tail_list[i]), batch_size=len(true_tail_list[i]),collate_fn=collate_en, shuffle=True, drop_last=True)
        

        for j, xx in enumerate((dataloader_en)): 
            #energy.eval()
            #print('-----------------------------')
            xx = util.move_to_cuda(xx)
            entity_tensor = energy.module.predict_ent_embedding(**xx)
            p_score =  torch.matmul(hr_vector[i],torch.transpose(entity_tensor, 0, 1))
            p_score_exp = torch.exp(p_score)#1024*1
            p_score_exp = p_score_exp.squeeze()
        all_p_score.append(p_score_exp)
    return all_p_score


def find_path_length(data_label):
    true_tails = data_label['tail']
    true_heads = data_label['head']
    score1 = torch.zeros(len(true_heads),len(eneities_labels))
    
    for i in range(len(true_heads)):
        head = true_heads[i]
        tail = true_tails[i]
        list_tail = graph_stru[head]['2']
        for tail in list_tail:
            k = eneities_labels.index(tail)
            score1[i][k] = 1
    return score1

            

    
def eva(batch_size = 256,
          num_false_neg = 3,
          num_hard_neg = 3,
          data_file_path = 'hasa/data/benchmarks/',
          Hasa_model = 'HaSa_OutHard',
          pretrain_model = 'sentence-transformers/all-mpnet-base-v2',
          input_dim = 768,
          em_dim = 500,
          data_name = 'FB15K237_L/',
          model_file=None,):
    


    device = torch.device('cuda')

    #if Hasa_model == 'HaSa':
        #print(Hasa_model)
    energy = HaSa(pretrain_model,input_dim,em_dim).to(device)
    loss_energy = InBatch_hard_NCEloss(batch_size = batch_size,hard_size = num_hard_neg,false_negative_size = num_false_neg)

    #if Hasa_model == 'HaSa_Hard_Bias':
        #print(Hasa_model)
        #energy = HaSa(pretrain_model,input_dim,em_dim).to(device)
        #loss_energy = Bert_batch_bias_hard(batch_size = batch_size,false_negative_size = 4)
        
    #if Hasa_model == 'HaSa_wohard_Bias':
        #print(Hasa_model)
        #energy =HaSa(pretrain_model,input_dim,em_dim).to(device)
        #loss_energy = Bert_batch_bias_wo_hard(batch_size = batch_size,false_negative_size = 4)


    triplet_reader = TripleLoader( data_file_path+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('test2name.csv')
    test_dataset = Dataset(examples = examples)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size,collate_fn=collate, shuffle=False, drop_last=True)



    dic_triplet = dic_true_triple(data_name)

    entity_reader = TripleLoader( 'hasa/data/benchmarks/'+data_name)
    eneities = entity_reader.get_entities('entity2name.csv')
    entity_dataset = Dataset(examples = eneities)
    dataloader_entity = DataLoader(entity_dataset, batch_size=256,collate_fn=collate_en, shuffle=False, drop_last=False)

    eneities_labels = entity_reader.get_entities_label('entity2name.csv')


    energy = torch.nn.DataParallel(energy).cuda()

    if model_file == None:
        print('Error: no model')
    else:

        checkpoint = torch.load(model_file)
        print('Resuming from checkpoint at ckpts/nce.pth.tar...')

        energy.load_state_dict(checkpoint['energy'])

    energy.eval()
    
    entity_list = []
    mm_list = []
    mrr_list = []
    hit10_list = [] 
    hit1_list = []
    hit3_list = []
    for j, xx in enumerate(tqdm(dataloader_entity)): 
        xx = util.move_to_cuda(xx)
        entity_tensor = energy.module.predict_ent_embedding(**xx)
        entity_list.append(entity_tensor)
    entity_tensor_all = torch.cat(entity_list, dim=0)
    dic_embedding_bank = {eneities_labels[i]:entity_tensor_all[i] for i in range(len(eneities_labels))}
    

    print('evaluation')

    
    for i, x_label in enumerate(tqdm(dataloader_test)): 
        x = x_label[0]
        y = x_label[1]
        tail_true_id = true_entites(y,dic_triplet,eneities_labels)


        x = util.move_to_cuda(x)

        gx,_,_ = energy.module.predict_mapping_embedding(**x)

        
                
        target_score = []
        xx = {}
        xx['entity_token_ids'] = x['tail_token_ids']
        xx['entity_token_type_ids'] = x['tail_token_type_ids']
        xx['entity_token_mask'] = x['tail_mask']
        tail_tensor_all = energy.module.predict_ent_embedding(**xx)
        #print(tail_tensor_all1.shape)
        true_tails = y['tail']
        tail_list = [dic_embedding_bank[tail_name].unsqueeze(0) for tail_name in true_tails]
        #tail_tensor_all = torch.cat(tail_list, dim=0)
        #print(tail_tensor_all.shape)

        scores_tensor_positve = loss_energy.evaluate1(gx,tail_tensor_all)
        #print(scores_tensor_positve)
        for idx in range(scores_tensor_positve.size(0)):
            target = scores_tensor_positve[idx,idx].item()
            target_score.append(target)
            
        rank_tensor = torch.zeros(batch_size)
                
        scores_tensor = loss_energy.evaluate1(gx,entity_tensor_all)

        for idx in range(scores_tensor.size(0)):
            #k = 
            target = round(target_score[idx],4)+0.001
            scores_tensor[idx][tail_true_id[idx]] = 0
            a = scores_tensor[idx]>target
            num_large = torch.count_nonzero(a).item()
            #print(idx)
            #print(num_large)
            rank_tensor[idx] = num_large

        b = rank_tensor<1
        #print(torch.count_nonzero(b).item()/batch_size)
        hit1_list.append(torch.count_nonzero(b).item()/batch_size)
        b = rank_tensor<3
        #print(torch.count_nonzero(b).item()/batch_size)
        hit3_list.append(torch.count_nonzero(b).item()/batch_size)
        b = rank_tensor<10
        #print(torch.count_nonzero(b).item()/batch_size)
        hit10_list.append(torch.count_nonzero(b).item()/batch_size)
        #print(torch.mean(rank_tensor))
        mm_list.append(torch.mean(rank_tensor))
        mrr_list.append(torch.mean(1/(rank_tensor+1)))
        if i>10:
            break

    print('mrr:',np.mean(mrr_list))
    print('mr:',np.mean(mm_list))
    print('@1:',np.mean(hit1_list))
    print('@3:',np.mean(hit3_list))
    print('@10:',np.mean(hit10_list))

