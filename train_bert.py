#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:13:05 2023

@author: hz
"""




import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torch import optim
from model_bert import HaSa,HaSa_Hard_Bias
#from model_robert import HaSa,HaSa_Hard_Bias
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
torch.manual_seed(40)


def embedding_bank(energy,entites_dataset_all,eneities_labels):
    #energy.eval()
    dic_embedding_bank = {}
    dataloader_entity = DataLoader(entites_dataset_all, batch_size=1000,collate_fn=collate_en, shuffle=False, drop_last=False)
    entity_list = []
    for j, xx_e in enumerate(dataloader_entity): 
        

        xx_e = util.move_to_cuda(xx_e)
        entity_tensor = energy.module.predict_ent_embedding(**xx_e)
        #entity_tensor = energy.predict_ent_embedding(**xx_e)
        entity_list.append(entity_tensor)
    entity_tensor_all = torch.cat(entity_list, dim=0)
    dic_embedding_bank = {eneities_labels[i]:entity_tensor_all[i] for i in range(len(eneities_labels))}
    
    return dic_embedding_bank
    


def train_bert(batch_size = 256,
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
    
    print('plus:',plus)
    


    triplet_reader = TripleLoader( data_file_path+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('train2name.csv')
    G= Graph_KG( data_file_path+data_name+'BERT_a/').graph_generator('train2name.csv')  

    dataloader = DataLoader(Dataset(examples = examples), batch_size=batch_size,collate_fn=collate, pin_memory=True, shuffle=True, drop_last=True)

    examples_valid = triplet_reader.get_examples('valid2name.csv')
    dataloader_valid = DataLoader(Dataset(examples = examples_valid), batch_size=batch_size,collate_fn=collate, shuffle=False, drop_last=True)

    entity_reader = TripleLoader( data_file_path+data_name)
    eneities = entity_reader.get_entities('entity2name.csv')
    eneities_label = entity_reader.get_entities_label('entity2name.csv')
    print('making graph')
    graph_stru = util.ego_graph_dic_all(eneities_label, G,length = 3)
    print('graph down---------------------------')
    entity_dataset = Dataset(examples = eneities)
    dic_triplet = util.dic_true_triple(data_name,'train2name.csv')
    #--------------------------------------------------
    device = torch.device('cuda')
    if Hasa_model == 'HaSa':
        print(Hasa_model)
        energy = HaSa(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = NegativeSampleHard(batch_size,num_false_neg,num_hard_neg,entity_dataset,
                                             eneities_label,graph_stru,dic_triplet)
        loss_energy = InBatch_hard_loss(batch_size = batch_size,hard_size = num_hard_neg,
                                        false_negative_size = num_false_neg,tau = tau,plus = plus)
        #loss_energy = InBatch_hard_NCEloss(batch_size = batch_size,hard_size = num_hard_neg,
                                        #false_negative_size = num_false_neg,tau = tau,plus = plus)

    if Hasa_model == 'HaSa_Hard_Bias':
        print(Hasa_model)
        energy = HaSa_Hard_Bias(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = NegativeHard_Bias(batch_size,num_hard_neg,entity_dataset,
                                            eneities_label,dic_triplet)
        loss_energy = Bert_batch_bias_hard(batch_size = batch_size,hard_size = num_hard_neg,plus = plus)
        
    if Hasa_model == 'HaSa_wohard_Bias':
        print(Hasa_model)
        energy = HaSa_Hard_Bias(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = Negative_Bias_naive(batch_size,num_hard_neg,entity_dataset,
                                            eneities_label,dic_triplet)
        loss_energy = Bert_batch_bias_wo_hard(batch_size = batch_size,hard_size = num_hard_neg,plus = plus)
    


    #optim_energy = optim.AdamW(energy.parameters(),
                                     #lr=learn_rate,weight_decay=1e-4)
    optim_energy = optim.AdamW(energy.parameters(),lr=learn_rate)
    if schedule_with_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer=optim_energy,
                                                   num_warmup_steps=100,num_training_steps=epoch_step*1000)

    loss_list = []
    loss_list_valid = []
    energy = torch.nn.DataParallel(energy).cuda()
    
    if Resuming:
        

        print('Resuming from checkpoint at ckpts/nce.pth.tar...')
        checkpoint = torch.load(loading_model)


        energy.load_state_dict(checkpoint['energy'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        
        start_epoch = 0
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



            #print(len(hard_labels))
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
            #loss,score = loss_energy(gx,zh,zt,ze,prob,f_prob)
            #print(loss)
            if np.isnan(loss.item()):
                print(loss)
                continue

            optim_energy.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(energy.parameters(), 5)
            optim_energy.step()
            
            if schedule_with_warmup:
                scheduler.step()


            if i%200 == 0:
                energy.eval()
                print(loss.detach().cpu().clone().numpy())
                loss_list.append(loss.detach().cpu().clone().numpy())
                results, metrics = loss_energy.prediction(gx,zh,zt)
                print(results)
                print(metrics)
                print(score)


            if i%500 == 0:
                print('valid_data')
                for j,x_v in enumerate(dataloader_valid):
                    energy.eval()
                    x_v = util.move_to_cuda(x_v[0])
                    gx,zh,zt = energy.module.predict_mapping_embedding(**x_v)

                    results, metrics = loss_energy.prediction(gx,zh,zt)
                    loss_list_valid.append(metrics['mrr'])
                    print(metrics)
                    if j>2:
                        break
                #loss_valid,score = loss_energy(gx,zh,zt,ze)

                if np.mean(loss_list_valid)> best_loss_valid:
                    best_loss_valid = np.mean(loss_list_valid)

                state = {'energy': energy.state_dict(),'value': loss,'epoch': epoch,}
                torch.save(state,output_file_path+'bestmodel_1e4.pth.tar')  


        state = {'energy': energy.state_dict(),'value': loss,'epoch': epoch,}
        torch.save(state, output_file_path+'bestmodel_1e4_'+str(epoch+1)+'.pth.tar')  



def train_sample(batch_size = 256,
          num_false_neg = 3,
          num_hard_neg = 3,
          data_file_path = 'hasa/data/benchmarks/',
          output_file_path = 'nfs_scratch/model_sample/FB/',
          Hasa_model = 'HaSa_OutHard',
          pretrain_model = 'sentence-transformers/all-mpnet-base-v2',
          input_dim = 768,
          em_dim = 500,
          epoch_step = 3,
          tau = 1e-4,
          data_name = 'FB15K237_L/',
          Resuming=False,
          loading_model=None,
          schedule_with_warmup=False,
          plus = True):
    


    triplet_reader = TripleLoader( data_file_path+data_name+'BERT_a/')
    examples = triplet_reader.get_examples('train2name_sample.csv')
    G= Graph_KG( data_file_path+data_name+'BERT_a/').graph_generator('train2name_sample.csv')  

    dataloader = DataLoader(Dataset(examples = examples), batch_size=batch_size,collate_fn=collate, pin_memory=True, shuffle=True, drop_last=True)

    examples_valid = triplet_reader.get_examples('valid2name.csv')
    dataloader_valid = DataLoader(Dataset(examples = examples_valid), batch_size=batch_size,collate_fn=collate, shuffle=False, drop_last=True)

    entity_reader = TripleLoader( data_file_path+data_name)
    eneities = entity_reader.get_entities('entity2name.csv')
    eneities_label = entity_reader.get_entities_label('entity2name.csv')
    print('making graph')
    graph_stru = util.ego_graph_dic_all(eneities_label, G,length = 3)
    print('graph down---------------------------')
    entity_dataset = Dataset(examples = eneities)
    dic_triplet = util.dic_true_triple(data_name,'train2name_sample.csv')
    #--------------------------------------------------
    device = torch.device('cuda')
    if Hasa_model == 'HaSa':
        print(Hasa_model)
        energy = HaSa(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = NegativeSampleHard(batch_size,num_false_neg,num_hard_neg,entity_dataset,
                                             eneities_label,graph_stru,dic_triplet)
        loss_energy = InBatch_hard_NCEloss(batch_size = batch_size,hard_size = num_hard_neg,
                                        false_negative_size = num_false_neg,tau = tau,plus = plus)

    if Hasa_model == 'HaSa_Hard_Bias':
        print(Hasa_model)
        energy = HaSa_Hard_Bias(pretrain_model,input_dim,em_dim).to(device)
        negative_sample = NegativeHard_Bias(batch_size,num_hard_neg,entity_dataset,
                                            eneities_label,dic_triplet)
        loss_energy = Bert_batch_bias_hard(batch_size = batch_size,hard_size = num_hard_neg,plus = plus)
    


    optim_energy = optim.AdamW(energy.parameters(),
                                     lr=2e-5,weight_decay=1e-4)
    if schedule_with_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer=optim_energy,
                                                   num_warmup_steps=100,num_training_steps=epoch_step*1000)

    loss_list = []
    loss_list_valid = []
    energy = torch.nn.DataParallel(energy).cuda()
    
    if Resuming:
        


        print('Resuming from checkpoint at ckpts/nce.pth.tar...')
        checkpoint = torch.load(loading_model)


        energy.load_state_dict(checkpoint['energy'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        
        start_epoch = 0
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
                x.update(x_f)
            else:
                prob,x_hard,hard_labels =negative_sample.neg_sampling(energy,xy,embedding_bank_, tail_true_id)


            energy.train()
            x_hard = util.move_to_cuda(x_hard)
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
            else:
                loss,score = loss_energy(gx,zh,zt,ze_hard,prob)
            #loss,score = loss_energy(gx,zh,zt,ze,prob,f_prob)
            #print(loss)
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
                print(loss.detach().cpu().clone().numpy())
                loss_list.append(loss.detach().cpu().clone().numpy())
                results, metrics = loss_energy.prediction(gx,zh,zt)
                print(results)
                print(metrics)
                print(score)


            if i%500 == 0:
                print('valid_data')
                for j,x_v in enumerate(dataloader_valid):
                    energy.eval()
                    x_v = util.move_to_cuda(x_v[0])
                    gx,zh,zt = energy.module.predict_mapping_embedding(**x_v)

                    results, metrics = loss_energy.prediction(gx,zh,zt)
                    loss_list_valid.append(metrics['mrr'])
                    print(metrics)
                    if j>2:
                        break
                #loss_valid,score = loss_energy(gx,zh,zt,ze)

                if np.mean(loss_list_valid)> best_loss_valid:
                    best_loss_valid = np.mean(loss_list_valid)

                state = {'energy': energy.state_dict(),'value': loss,'epoch': epoch,}
                torch.save(state,output_file_path+'bestmodel_1e4_'+str(epoch+1)+str(i)+'.pth.tar')  


        state = {'energy': energy.state_dict(),'value': loss,'epoch': epoch,}
        torch.save(state, output_file_path+'bestmodel_1e4_'+str(epoch+1)+'.pth.tar') 



