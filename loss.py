#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:17:45 2022

@author: user1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import collections

def prediction_(s, v_h,v_t):


    v_h_t = torch.cat((v_t,v_h), 0)
    p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))
    p_score_exp = torch.exp(p_score)#1024*1
    p_score_exp = p_score_exp.squeeze()
        #print(p_score_exp)
    batch_sorted_score, batch_sorted_indices = torch.sort(p_score_exp, dim=-1, descending=True)
        #print(batch_sorted_score.shape)#64*15000
    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    ranks = []
    for idx in range(batch_sorted_score.size(0)):
        rank_idx = batch_sorted_indices[idx]
        cur_rank = (rank_idx==idx).nonzero().item()


            # 0-based -> 1-based
        cur_rank += 1
        mean_rank += cur_rank
        mrr += 1.0 / cur_rank
        hit1 += 1 if cur_rank <= 1 else 0
        hit3 += 1 if cur_rank <= 3 else 0
        hit10 += 1 if cur_rank <= 10 else 0
        ranks.append(cur_rank)
    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / batch_sorted_score.size(0), 4) for k, v in metrics.items()}

        #print(p_bias_score_exp.shape)
    num_j = []
    for i in range(len(p_score_exp)):
        pp = p_score_exp[i][i].cpu().detach().numpy()
        nn =  p_score_exp[i].cpu().detach().numpy()
        nn = np.sort(nn)   
            #print(pp)
            #print(nn[-10:])
        for j in range(len(nn)):
            if pp>nn[len(nn)-1-j]:
                num_j.append(j+1)
                        
                break
    result = collections.Counter(num_j).most_common()
    return result,metrics

    
def prediction_train_(s, v_h,v_t,v_hard,v_e):

    v_h_t = torch.cat((v_t,v_h,v_hard,v_e), 0)
    p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))
    p_score_exp = torch.exp(p_score)#1024*1
    p_score_exp = p_score_exp.squeeze()
        #print(p_score_exp)
    batch_sorted_score, batch_sorted_indices = torch.sort(p_score_exp, dim=-1, descending=True)
        #print(batch_sorted_score.shape)#64*15000
    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    ranks = []
    for idx in range(batch_sorted_score.size(0)):
        rank_idx = batch_sorted_indices[idx]
        cur_rank = (rank_idx==idx).nonzero().item()


            # 0-based -> 1-based
        cur_rank += 1
        mean_rank += cur_rank
        mrr += 1.0 / cur_rank
        hit1 += 1 if cur_rank <= 1 else 0
        hit3 += 1 if cur_rank <= 3 else 0
        hit10 += 1 if cur_rank <= 10 else 0
        ranks.append(cur_rank)
    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / batch_sorted_score.size(0), 4) for k, v in metrics.items()}

        #print(p_bias_score_exp.shape)
    num_j = []
    for i in range(len(p_score_exp)):
        pp = p_score_exp[i][i].cpu().detach().numpy()
        nn =  p_score_exp[i].cpu().detach().numpy()
        nn = np.sort(nn)   

        for j in range(len(nn)):
            if pp>nn[len(nn)-1-j]:
                num_j.append(j+1)
                        
                break
    result = collections.Counter(num_j).most_common()
        
    return result,metrics

def evaluate_(s, all_e):

    v_h_t = all_e
    p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))
    p_score_exp = torch.exp(p_score)#1024*1
    p_score_exp = p_score_exp.squeeze()
    return p_score_exp

class InBatch_hard_loss(nn.Module):
    
    def __init__(self,batch_size = 100,hard_size = 5,false_negative_size = 4,tau = 0.05, plus = True):
        super(InBatch_hard_loss, self).__init__()
        
        self.batch_size = batch_size 
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.negative_size = false_negative_size 
        self.hard_size = hard_size
        self.tau = tau
        self.plus = plus
        
    def forward(self,s,v_h,v_t,v_f,v_e_hard,prob,f_prob):

        a,b = v_h.shape
        labels = torch.arange(a).to(v_h.device)

        v_h_t = torch.cat((v_t, v_h,v_e_hard), 0)
        p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))
        f_score =  torch.matmul(s.view(a,1,b),torch.transpose(v_f.view(a,self.negative_size,b), 1, 2))
        f_score = f_score.squeeze()
            
        prob = prob.to(p_score.device).detach()
        _,num_neg = v_h_t.shape

        f_prob = f_prob.to(f_score.device).detach()
        #f_score = f_score1 + f_prob
        
        de2 = torch.exp(f_score)*f_prob
        #de2 = torch.exp(f_score1)
        de2_sum = self.tau*de2.sum(1)
        p_score -= torch.zeros(p_score.size()).fill_diagonal_(1).to(p_score.device)
        nu = torch.exp(p_score).diagonal(0)
        p_score_exp = torch.exp(p_score)
        #print(p_score_exp.shape)
        #print(p_score_exp)
        
        de1 = p_score_exp*prob
        #de1 = p_score_exp
        de1_sum = de1.sum(1)-nu
        #print(de1_sum-de2_sum)
        de = F.threshold(de1_sum-de2_sum,0,1,inplace=True)
        #de = de1_sum
        #print(de)
 
        loss = -1*torch.log(nu/(nu+(num_neg/(1-self.tau))*de)).mean()
    
        if self.plus:
            labels = torch.arange(a).to(v_h.device)
            loss += self.criterion(p_score[:, :a].t(), labels)
        #print(loss)

        return loss, p_score[:, :a] 
        #return loss, de1


    
    def prediction_train(self, s, v_h,v_t,v_hard,v_e):
        return prediction_train(s, v_h,v_t,v_hard,v_e)
    
    def prediction(self, s, v_h,v_t):
        return prediction_(s, v_h,v_t)

    def evaluate1(self, s, all_e):
     
        return evaluate_(s, all_e)


    



class Bert_batch_bias_hard(nn.Module):
    
    def __init__(self,batch_size = 100,hard_size = 5,plus = True):
        super(Bert_batch_bias_hard, self).__init__()
        
        self.batch_size = batch_size 
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.hard_size = hard_size
        self.plus = plus
        
    def forward(self,s,v_h,v_t,v_e_hard,prob):
  
        a,b = v_h.shape
        
        #v_h_t = v_t


        labels = torch.arange(a).to(v_h.device)

        v_h_t = torch.cat((v_t, v_h,v_e_hard), 0)
        p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))

        #print(p_score.shape)
            
        prob = prob.to(p_score.device).detach()
        _,num_neg = v_h_t.shape

 
        p_score -= torch.zeros(p_score.size()).fill_diagonal_(1).to(p_score.device)
        nu = torch.exp(p_score).diagonal(0)
        p_score_exp = torch.exp(p_score)
        #print(p_score_exp.shape)
        #print(p_score_exp)
        
        de1 = p_score_exp*prob
        #de1 = p_score_exp
        de1_sum = de1.sum(1)-nu
        #print(de1_sum-de2_sum)
        de = de1_sum
        #print(de)
 
        loss = -1*torch.log(nu/(nu+(num_neg)*de)).mean()
        if self.plus:
            labels = torch.arange(a).to(v_h.device)
            loss += self.criterion(p_score[:, :a].t(), labels)

        return loss, p_score[:, :a] 


    
    def prediction_train(self, s, v_h,v_t,v_hard,v_e):
        return prediction_train(s, v_h,v_t,v_hard,v_e)
    
    def prediction(self, s, v_h,v_t):
        return prediction_(s, v_h,v_t)


    def evaluate1(self, s, all_e):
     
        return evaluate_(s, all_e)



class InBatch_hard_NCEloss(nn.Module):
    
    def __init__(self,batch_size = 100,hard_size = 5,false_negative_size = 4,tau = 0.05, plus = True):
        super(InBatch_hard_NCEloss, self).__init__()
        
        self.batch_size = batch_size 
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.negative_size = false_negative_size 
        self.hard_size = hard_size
        self.tau = tau
        self.plus = plus
        
    def forward(self,s,v_h,v_t,v_f,v_e_hard,prob,f_prob):

        a,b = v_h.shape
        labels = torch.arange(a).to(v_h.device)

        v_h_t = torch.cat((v_t, v_h,v_e_hard, v_f), 0)
        p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))
        f_score =  torch.matmul(s.view(a,1,b),torch.transpose(v_f.view(a,self.negative_size,b), 1, 2))
        f_score = f_score.squeeze()
            
        prob = prob.to(p_score.device).detach()
        _,num_neg = v_h_t.shape

        f_prob = f_prob.to(f_score.device).detach()
        f_score_sig = torch.sigmoid(-1*f_score)
        #f_score = f_score1 + f_prob
        
        de2 = torch.log(f_score_sig)*f_prob
        de2_sum = self.tau*de2.sum(1)
        
        p_score -= torch.zeros(p_score.size()).fill_diagonal_(1).to(p_score.device)
        nu = p_score.diagonal(0)
        nu_sig = torch.sigmoid(nu)
        
        p_score_sig = torch.sigmoid(-1*p_score)
        #print(p_score_exp.shape)
        #print(p_score_exp)
        
        de1 = torch.log(p_score_sig)*prob
        #de1 = p_score_exp
        de1_sum = de1.sum(1)-torch.log(torch.sigmoid(-1*nu))
        #print(de1_sum-de2_sum)
        #de = F.threshold(de1_sum-de2_sum,0,1,inplace=True)
        de = de1_sum-de2_sum
        #print(de)
 
        loss = -1*(torch.log(nu_sig)+(1/(1-self.tau))*de).mean()
    

        return loss, p_score[:, :a] 
        #return loss, de1
        
    def prediction_train(self, s, v_h,v_t,v_hard,v_e):
        return prediction_train(s, v_h,v_t,v_hard,v_e)
    
    def prediction(self, s, v_h,v_t):
        return prediction_(s, v_h,v_t)


    def evaluate1(self, s, all_e):
     
        return evaluate_(s, all_e)
    
class Bert_batch_bias_wo_hard(nn.Module):
    
    def __init__(self,batch_size = 100,hard_size = 5,plus = True):
        super(Bert_batch_bias_wo_hard, self).__init__()
        
        self.batch_size = batch_size 
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.hard_size = hard_size
        self.plus = plus
        
    def forward(self,s,v_h,v_t,v_e_hard):
  
        a,b = v_h.shape
        
        #v_h_t = v_t


        labels = torch.arange(a).to(v_h.device)

        v_h_t = torch.cat((v_t, v_h,v_e_hard), 0)
        p_score =  torch.matmul(s,torch.transpose(v_h_t, 0, 1))
 
        p_score -= torch.zeros(p_score.size()).fill_diagonal_(1).to(p_score.device)
        loss = self.criterion(p_score, labels)
        if self.plus:
            labels = torch.arange(a).to(v_h.device)
            loss += self.criterion(p_score[:, :a].t(), labels)

        return loss, p_score[:, :a] 


    
    def prediction_train(self, s, v_h,v_t,v_hard,v_e):
        return prediction_train(s, v_h,v_t,v_hard,v_e)
    
    def prediction(self, s, v_h,v_t):
        return prediction_(s, v_h,v_t)


    def evaluate1(self, s, all_e):
     
        return evaluate_(s, all_e)