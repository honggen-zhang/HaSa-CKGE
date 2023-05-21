#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 15:57:37 2022

@author: user1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
# ------------------------------
# ENERGY-BASED MODEL
# ------------------------------

class HaSa(nn.Module):
    def __init__(self,pretrained_model, input_dim, embedding_dim):
        super(HaSa, self).__init__()
        # The normalizing constant logZ(θ)        
        #self.c = nn.Parameter(torch.tensor([1.], requires_grad=True))
        #self.g = 'indetity'
        self.pretrained_model = pretrained_model
        self.dim = embedding_dim
        self.input_dim = input_dim
        #input 768
        #self.bilinear = nn.Bilinear(self.dim, self.dim, self.dim)

        self.bert =  AutoModel.from_pretrained(self.pretrained_model)
        #self.bert =  AutoModel.from_pretrained("bert-base-uncased")
        self.dense = nn.Sequential(nn.Linear(self.input_dim,self.dim,bias = True),
                                   nn.LayerNorm(self.dim,eps=1e-12, elementwise_affine=True),
                                   nn.Dropout(p=0.1,inplace = False),)
        #self.mapping = nn.Sequential(nn.Bilinear(self.dim, self.dim, self.dim),
                                     #nn.Linear(self.dim,self.dim,bias = True),
                                     #nn.LayerNorm(self.dim,eps=1e-12, elementwise_affine=True),)
        #self.dense = nn.Identity()

        self.gru = nn.GRU(self.dim, self.dim, 1,batch_first = True)
        

    def forward(self, r_token_ids, r_mask, r_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids, 
                entity_token_ids, entity_token_mask,entity_token_type_ids,
                entity_token_ids_hard, entity_token_mask_hard,entity_token_type_ids_hard,**kwargs) -> dict:
        

        output_h = self.bert(input_ids = head_token_ids,attention_mask = head_mask)
        z_h  = output_h.pooler_output
        zh = self.dense(z_h)
        
        output_r = self.bert(input_ids = r_token_ids,attention_mask = r_mask)
        z_r  = output_r.pooler_output
        zr = self.dense(z_r)
        
        output_t = self.bert(input_ids = tail_token_ids,attention_mask = tail_mask)
        z_t  = output_t.pooler_output
        zt = self.dense(z_t)
        a,b = zh.shape
        
        output_e = self.bert(input_ids = entity_token_ids,attention_mask = entity_token_mask)
        z_e  = output_e.pooler_output
        ze = self.dense(z_e)
        
        output_e_hard = self.bert(input_ids = entity_token_ids_hard,attention_mask = entity_token_mask_hard)
        z_e_hard  = output_e_hard.pooler_output
        ze_hard = self.dense(z_e_hard)
        

        z_hr = torch.cat((zh.view(a,1,b), zr.view(a,1,b)), 1)
        self.gru.flatten_parameters()
        aggregate,hn = self.gru(z_hr)
        g_x = hn.transpose(0,1).squeeze()
        #g_x = self.bilinear(zh,zr)
        #g_x = g_x.squeeze()


        return g_x,zh,zt,ze,ze_hard
        
    
    @torch.no_grad()
    def predict_ent_embedding(self, entity_token_ids, entity_token_mask,entity_token_type_ids)-> dict:
        
        output = self.bert(input_ids = entity_token_ids,attention_mask = entity_token_mask)
        z_  = output.pooler_output
        #print(z_.shape)
        z = self.dense(z_)

        

        return z.detach()
    @torch.no_grad()
    def predict_mapping_embedding(self, r_token_ids, r_mask, r_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,**kwargs) -> dict:


        output_h = self.bert(input_ids = head_token_ids,attention_mask = head_mask)
        z_h  = output_h.pooler_output
        zh = self.dense(z_h)
        
        
        output_t = self.bert(input_ids = tail_token_ids,attention_mask = tail_mask)
        z_t  = output_t.pooler_output
        zt = self.dense(z_t)
        
        output_r = self.bert(input_ids = r_token_ids,attention_mask = r_mask)
        z_r  = output_r.pooler_output
        zr = self.dense(z_r)
        
        a,b = zh.shape
        #a,b = zh.shape
        #z = z.view(int(a/2),2,b)
        #z_hr = z[:,0:2,:]
        #z_hr_f = z_hr.view(int(a/3),2*b)
        #g_x  = self.mapping(z_hr_f)
        #print(z_hr.shape)
        z_hr = torch.cat((zh.view(a,1,b), zr.view(a,1,b)), 1)
        self.gru.flatten_parameters()
        aggregate,hn = self.gru(z_hr)
        g_x = hn.transpose(0,1).squeeze()
        #g_x = self.bilinear(zh,zr)
        #g_x = g_x.squeeze()

        return g_x.detach(), zh.squeeze().detach(), zt.detach()

    

class HaSa_Hard_Bias(nn.Module):
    def __init__(self,pretrained_model, input_dim, embedding_dim):
        super(HaSa_Hard_Bias, self).__init__()
        self.pretrained_model = pretrained_model
        self.dim = embedding_dim
        self.input_dim = input_dim


        self.bert =  AutoModel.from_pretrained(self.pretrained_model)
        #self.bert =  AutoModel.from_pretrained("bert-base-uncased")
        self.dense = nn.Sequential(nn.Linear(self.input_dim,self.dim,bias = True),
                                   nn.LayerNorm(self.dim,eps=1e-12, elementwise_affine=True),
                                   nn.Dropout(p=0.1,inplace = False),)


        self.gru = nn.GRU(self.dim, self.dim, 1,batch_first = True)
        
 
    


    def forward(self, r_token_ids, r_mask, r_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids, 
                entity_token_ids_hard, entity_token_mask_hard,entity_token_type_ids_hard,**kwargs) -> dict:
        

        output_h = self.bert(input_ids = head_token_ids,attention_mask = head_mask)
        z_h  = output_h.pooler_output
        zh = self.dense(z_h)
        
        output_r = self.bert(input_ids = r_token_ids,attention_mask = r_mask)
        z_r  = output_r.pooler_output
        zr = self.dense(z_r)
        
        output_t = self.bert(input_ids = tail_token_ids,attention_mask = tail_mask)
        z_t  = output_t.pooler_output
        zt = self.dense(z_t)
        a,b = zh.shape
        
        output_e_hard = self.bert(input_ids = entity_token_ids_hard,attention_mask = entity_token_mask_hard)
        z_e_hard  = output_e_hard.pooler_output
        ze_hard = self.dense(z_e_hard)
        

        z_hr = torch.cat((zh.view(a,1,b), zr.view(a,1,b)), 1)
        self.gru.flatten_parameters()
        aggregate,hn = self.gru(z_hr)
        g_x = hn.transpose(0,1).squeeze()
        #g_x = self.bilinear(zh,zr)
        #g_x = g_x.squeeze()


        return g_x,zh,zt,ze_hard
    

    
    @torch.no_grad()
    def predict_ent_embedding(self, entity_token_ids, entity_token_mask,entity_token_type_ids)-> dict:
        
        output = self.bert(input_ids = entity_token_ids,attention_mask = entity_token_mask)
        z_  = output.pooler_output
        #print(z_.shape)
        z = self.dense(z_)

        

        return z.detach()
    @torch.no_grad()
    def predict_mapping_embedding(self, r_token_ids, r_mask, r_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,**kwargs) -> dict:


        output_h = self.bert(input_ids = head_token_ids,attention_mask = head_mask)
        z_h  = output_h.pooler_output
        zh = self.dense(z_h)
        
        
        output_t = self.bert(input_ids = tail_token_ids,attention_mask = tail_mask)
        z_t  = output_t.pooler_output
        zt = self.dense(z_t)
        
        output_r = self.bert(input_ids = r_token_ids,attention_mask = r_mask)
        z_r  = output_r.pooler_output
        zr = self.dense(z_r)
        
        a,b = zh.shape
        #a,b = zh.shape
        #z = z.view(int(a/2),2,b)
        #z_hr = z[:,0:2,:]
        #z_hr_f = z_hr.view(int(a/3),2*b)
        #g_x  = self.mapping(z_hr_f)
        #print(z_hr.shape)
        z_hr = torch.cat((zh.view(a,1,b), zr.view(a,1,b)), 1)
        self.gru.flatten_parameters()
        aggregate,hn = self.gru(z_hr)
        g_x = hn.transpose(0,1).squeeze()
        #g_x = self.bilinear(zh,zr)
        #g_x = g_x.squeeze()

        return g_x.detach(), zh.squeeze().detach(), zt.detach()
       

    
    @torch.no_grad()
    def predict_ent_embedding(self, entity_token_ids, entity_token_mask,entity_token_type_ids)-> dict:
        
        output = self.bert(input_ids = entity_token_ids,attention_mask = entity_token_mask)
        z_  = output.pooler_output
        #print(z_.shape)
        z = self.dense(z_)

        

        return z.detach()
    @torch.no_grad()
    def predict_mapping_embedding(self, r_token_ids, r_mask, r_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,**kwargs) -> dict:


        output_h = self.bert(input_ids = head_token_ids,attention_mask = head_mask)
        z_h  = output_h.pooler_output
        zh = self.dense(z_h)
        
        
        output_t = self.bert(input_ids = tail_token_ids,attention_mask = tail_mask)
        z_t  = output_t.pooler_output
        zt = self.dense(z_t)
        
        output_r = self.bert(input_ids = r_token_ids,attention_mask = r_mask)
        z_r  = output_r.pooler_output
        zr = self.dense(z_r)
        
        a,b = zh.shape
        zh = zh.view(a,1,b)
        zr = zr.view(a,1,b)
        #z = F.normalize(z,dim=1)
        z_hr = torch.cat((zh, zr), 1)
        #a,b = zh.shape
        #z = z.view(int(a/2),2,b)
        #z_hr = z[:,0:2,:]
        #z_hr_f = z_hr.view(int(a/3),2*b)
        #g_x  = self.mapping(z_hr_f)
        #print(z_hr.shape)
        aggregate,hn = self.gru(z_hr)
        g_x = hn.transpose(0,1).squeeze()
        

        return g_x.detach(), zh.squeeze().detach(), zt.detach()

    




    
