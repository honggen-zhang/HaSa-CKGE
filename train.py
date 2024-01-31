#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:13:05 2023

@author: hz
"""



import logging
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch import optim
from model import HaSa,HaSa_Hard_Bias
from loss import InBatch_hard_loss,Bert_batch_bias_hard,InBatch_hard_NCEloss,Bert_batch_bias_wo_hard
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from TripleLoader import TripleLoader,Dataset,collate,collate_en,Graph_KG
from negative_sample import NegativeSampleHard
from negative_sample_bias import NegativeHard_Bias,Negative_Bias_naive
from time import sleep
from tqdm import tqdm
import operator
import numpy as np
import util
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from random import sample
np.random.seed(40)
torch.manual_seed(40)
logging.basicConfig(level=logging.INFO) 


def embedding_bank(energy,entites_dataset_all,eneities_labels):
    #energy.eval()
    dic_embedding_bank = {}
    dataloader_entity = DataLoader(entites_dataset_all, batch_size=1000,collate_fn=collate_en, shuffle=False, drop_last=False)
    entity_list = []
    for j, xx_e in enumerate(dataloader_entity): 
        xx_e = util.move_to_cuda(xx_e)
        entity_tensor = energy.module.predict_ent_embedding(**xx_e)
        entity_list.append(entity_tensor)
    entity_tensor_all = torch.cat(entity_list, dim=0)
    dic_embedding_bank = {eneities_labels[i]:entity_tensor_all[i] for i in range(len(eneities_labels))}
    
    return dic_embedding_bank
    


def train(batch_size = 256,
          learn_rate = 2e-5,
          num_false_neg = 3,
          num_hard_neg = 3,
          data_file_path = 'hasa/data/benchmarks/',
          output_file_path = 'nfs_scratch/FB_model_tau/1e4/',
          Hasa_model = 'HaSa_OutHard',
          pretrain_model = 'sentence-transformers/all-mpnet-base-v2',
          input_dim = 768,
          em_dim = 500,
          epoch_step = 5,
          tau = 1e-4,
          data_name = 'FB15K237_L/',
          Resuming=False,
          loading_model=None,
          schedule_with_warmup=False,
          plus = True):
    
    logging.info('Using symtrical loss function: %s',plus)
    

    logging.info('Loading the tranning data...')
    triplet_reader = TripleLoader( data_file_path+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('train2name.csv')
    logging.info('Loading the graph...')
    G= Graph_KG( data_file_path+data_name+'BERT_a/').graph_generator('train2name.csv')  

    dataloader = DataLoader(Dataset(examples = examples), batch_size=batch_size,collate_fn=collate, pin_memory=True, shuffle=True, drop_last=True)

    examples_valid = triplet_reader.get_examples('valid2name.csv')
    dataloader_valid = DataLoader(Dataset(examples = examples_valid), batch_size=batch_size,collate_fn=collate, shuffle=False, drop_last=True)
    logging.info('Loading the entity and vectorize data...')
    entity_reader = TripleLoader( data_file_path+data_name)
    eneities = entity_reader.get_entities('entity2name.csv')
    eneities_label = entity_reader.get_entities_label('entity2name.csv')
    logging.info('Builing the ego graph...')
    graph_stru = util.ego_graph_dic_all(eneities_label, G,length = 3)
    logging.info('Builing the dictionary of query...')
    entity_dataset = Dataset(examples = eneities)
    dic_triplet = util.dic_true_triple(examples)
    #--------------------------------------------------
    device = torch.device('cuda')
    logging.info('Using model: %s',Hasa_model)
    if Hasa_model == 'HaSa':
        energy = HaSa(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = NegativeSampleHard(batch_size,num_false_neg,num_hard_neg,entity_dataset,
                                             eneities_label,graph_stru,dic_triplet)
        loss_energy = InBatch_hard_loss(batch_size = batch_size,hard_size = num_hard_neg,
                                        false_negative_size = num_false_neg,tau = tau,plus = plus)

    if Hasa_model == 'HaSa_Hard_Bias':
        energy = HaSa_Hard_Bias(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = NegativeHard_Bias(batch_size,num_hard_neg,entity_dataset,
                                            eneities_label,dic_triplet)
        loss_energy = Bert_batch_bias_hard(batch_size = batch_size,hard_size = num_hard_neg,plus = plus)
        
    if Hasa_model == 'HaSa_wohard_Bias':
        energy = HaSa_Hard_Bias(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = Negative_Bias_naive(batch_size,num_hard_neg,entity_dataset,
                                            eneities_label,dic_triplet)
        loss_energy = Bert_batch_bias_wo_hard(batch_size = batch_size,hard_size = num_hard_neg,plus = plus)
    


    #optim_energy = optim.AdamW(energy.parameters(),
                                     #lr=learn_rate,weight_decay=1e-4)
    optim_energy = optim.AdamW(energy.parameters(),
                                     lr=learn_rate)
    if schedule_with_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer=optim_energy,
                                                   num_warmup_steps=100,num_training_steps=epoch_step*1000)

    loss_list = []
    loss_list_valid = []
    energy = torch.nn.DataParallel(energy).cuda()
    
    if Resuming:
        
        logging.info('Resuming from checkpoint...')
        checkpoint = torch.load(loading_model)
        energy.load_state_dict(checkpoint['energy'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    logging.info('Initlial the entity embedding bank...')
    embedding_bank_ = embedding_bank(energy,entity_dataset,eneities_label)


    best_loss_valid = 0.1
    for epoch in range(start_epoch, start_epoch+epoch_step):

        for i, xy in enumerate(tqdm(dataloader)): 
            
            x = xy[0]
            y = xy[1]
            x = util.move_to_cuda(x)
            tail_true_id = util.true_entites(y,dic_triplet,eneities_label)
            if Hasa_model == 'HaSa':
                prob,f_prob,x_f, f_labels,x_hard,hard_labels=negative_sample.neg_sampling(energy,xy,embedding_bank_, tail_true_id)
                x_f = util.move_to_cuda(x_f)
                # extend the x with the false negative embedding
                x.update(x_f)

            if Hasa_model == 'HaSa_Hard_Bias':
                prob,x_hard,hard_labels =negative_sample.neg_sampling(energy,xy,embedding_bank_, tail_true_id)
                
            if Hasa_model == 'HaSa_wohard_Bias':
                x_hard,hard_labels =negative_sample.neg_sampling(energy,xy,embedding_bank_, tail_true_id)

            energy.train()
            x_hard = util.move_to_cuda(x_hard)
            # extend the x with the hard negative embedding
            x.update(x_hard)
            if Hasa_model == 'HaSa':
                gx,zh,zt,zf,ze_hard = energy(**x)
            else:
                gx,zh,zt,ze_hard = energy(**x)

            dic_bank_batch_head = {y['head'][i]:zh[i] for i in range(len(zh))}
            dic_bank_batch_tail = {y['tail'][i]:zt[i] for i in range(len(zh))}
            dic_bank_batch_e_hard = {hard_labels[i]:ze_hard[i] for i in range(len(ze_hard))}
            embedding_bank_.update(dic_bank_batch_head)
            embedding_bank_.update(dic_bank_batch_tail)
            embedding_bank_.update(dic_bank_batch_e_hard)
            if Hasa_model == 'HaSa':
                dic_bank_batch_f = {f_labels[i]:zf[i] for i in range(len(zf))}
                embedding_bank_.update(dic_bank_batch_f)
                loss,score = loss_energy(gx,zh,zt,zf,ze_hard,prob,f_prob)
            if Hasa_model == 'HaSa_Hard_Bias':
                loss,score = loss_energy(gx,zh,zt,ze_hard,prob)
            if Hasa_model == 'HaSa_wohard_Bias':
                loss,score = loss_energy(gx,zh,zt,ze_hard)

            if np.isnan(loss.item()):
                print(loss)
                continue

            optim_energy.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(energy.parameters(), 5)
            optim_energy.step()
            
            if schedule_with_warmup:
                scheduler.step()


            if i%100 == 0:
                energy.eval()
                loss_list.append(loss.detach().cpu().clone().numpy())
                results, metrics = loss_energy.prediction(gx,zh,zt)
                logging.info('training result: %s',metrics)



            if i%500 == 0:
                for j,x_v in enumerate(dataloader_valid):
                    energy.eval()
                    x_v = util.move_to_cuda(x_v[0])
                    gx,zh,zt = energy.module.predict_mapping_embedding(**x_v)

                    results, metrics = loss_energy.prediction(gx,zh,zt)
                    loss_list_valid.append(metrics['mrr'])
                    logging.info('valid result: %s',metrics)
                    if j>2:
                        break


                if np.mean(loss_list_valid)> best_loss_valid:
                    best_loss_valid = np.mean(loss_list_valid)

                state = {'energy': energy.state_dict(),'value': loss,'epoch': epoch,}
                torch.save(state,output_file_path+'bestmodel.pth.tar')  


        state = {'energy': energy.state_dict(),'value': loss,'epoch': epoch,}
        torch.save(state, output_file_path+'model_'+str(epoch+1)+'.pth.tar')  





