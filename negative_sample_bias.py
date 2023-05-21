import torch.nn as nn
import torch
import util
from random import sample
import numpy as np

class NegativeHard_Bias(object):
    
    def __init__(self,batch_size = 256, hard_size=5, entites_dataset_all=None,eneities_labels=None,dic_triplet = None):
        super(NegativeHard_Bias, self).__init__()

        self.negative_hard_size = hard_size
        self.entites_dataset_all = entites_dataset_all 
        self.eneities_labels = eneities_labels
        self.batch_size = batch_size
        self.dic_triplet = dic_triplet

        
    def hard_tails_em(self,em_bank,gx_input,y_input,batch_input_true_ids):

        entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        entity_tensor_all = torch.cat(entity_list, dim=0)

        p_score =  torch.matmul(gx_input,torch.transpose(entity_tensor_all, 0, 1))
        p_score_exp = torch.exp(p_score)#1024*1
        true_tails = y_input['tail']
        for i in range(len(true_tails)):
            k = self.eneities_labels.index(true_tails[i])
            p_score_exp[i][k] = 0
            other_true_ids= batch_input_true_ids[i]
            for j in other_true_ids:
                #k = other_true_ids[j]
                p_score_exp[i][j] = 0




        sorted_score, sorted_indices = torch.sort(p_score_exp, dim=-1, descending=True)

        entity_input_list_lable = []
        entites_dataset_all_token = []
        entites_dataset_all_mask = []
        entites_dataset_all_type = []
        for i in range(self.batch_size):
            for j in range(self.negative_hard_size):
                k = sorted_indices[i][j].item()
                #print(k)
                ent_label = self.eneities_labels[k]
                en_example = self.entites_dataset_all[k]
                entity_input_list_lable.append(ent_label)


                entites_dataset_all_token.append(torch.LongTensor(en_example['entity_token_ids']).unsqueeze(0))
                entites_dataset_all_mask.append(torch.LongTensor(en_example['entity_token_mask']).unsqueeze(0))
                entites_dataset_all_type.append(torch.LongTensor(en_example['entity_token_type_ids']).unsqueeze(0))

        entites_dataset_all_token_tensor = torch.cat((entites_dataset_all_token),0)
        entites_dataset_all_mask_tensor = torch.cat((entites_dataset_all_mask),0)
        entites_dataset_all_type_tensor = torch.cat((entites_dataset_all_type),0)

        return {'entity_token_ids_hard':entites_dataset_all_token_tensor,
        'entity_token_type_ids_hard':entites_dataset_all_type_tensor,
        'entity_token_mask_hard':entites_dataset_all_mask_tensor,},entity_input_list_lable
    
    




    def neg_sampling(self,energy, batch_input,em_bank, batch_input_true_ids):
        energy.eval()

        #entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        #entity_tensor_all = torch.cat(entity_list, dim=0)
        x_input = batch_input[0]
        y_input = batch_input[1]
        x_input = util.move_to_cuda(x_input)
        gx_input,_,_ = energy.module.predict_mapping_embedding(**x_input)
        #print(gx_input.shape)

        true_tails = y_input['tail']
        true_head = y_input['head']
        true_r= y_input['relation']
        
        dic_hard_tails, hard_tails =self.hard_tails_em(em_bank,gx_input,y_input,batch_input_true_ids)


        y_tails = y_input['tail']+y_input['head']+hard_tails
        entity_list = [em_bank[en_name].unsqueeze(0) for en_name in y_tails]
        entity_tensor_all = torch.cat(entity_list, dim=0)




        p_score =  torch.matmul(gx_input,torch.transpose(entity_tensor_all, 0, 1))
        #p_score_exp = torch.exp(p_score)#1024*1
        p_score_exp = nn.functional.normalize(p_score,dim = 1)
        for i in range(self.batch_size):
            #k = eneities_labels.index(true_tails[i])
            p_score_exp[i][i] = -10
            other_true_ids= batch_input_true_ids[i]
            for j in other_true_ids:
                false_tail = self.eneities_labels[j]
                try:
                    #k_list = [k for k in range(len(a)) if a[k] == 'b']
                    k = true_tails.index(false_tail)
                    p_score_exp[i][k] = -10
                except:
                    pass

 
        prob = nn.functional.softmax(p_score_exp,dim = 1)
        #print(prob)
        for l in range(self.batch_size):
            prob[l][l] = 1



        return prob,dic_hard_tails,hard_tails



    
class Negative_Bias_naive(object):
    
    def __init__(self,batch_size = 256, hard_size=5, entites_dataset_all=None,eneities_labels=None,dic_triplet = None):
        super(Negative_Bias_naive, self).__init__()

        self.negative_hard_size = hard_size
        self.entites_dataset_all = entites_dataset_all 
        self.eneities_labels = eneities_labels
        self.batch_size = batch_size
        self.dic_triplet = dic_triplet
        
    def hard_tails_em_naive(self,em_bank,y_input,batch_input_true_ids):


        entity_input_list_lable = []
        entites_dataset_all_token = []
        entites_dataset_all_mask = []
        entites_dataset_all_type = []
        random_k = np.random.randint(len(self.eneities_labels), size=(self.batch_size, self.negative_hard_size))
        for i in range(self.batch_size):
            for j in range(self.negative_hard_size):
                k = random_k[i][j]
                #print(k)
                ent_label = self.eneities_labels[k]
                en_example = self.entites_dataset_all[k]
                entity_input_list_lable.append(ent_label)


                entites_dataset_all_token.append(torch.LongTensor(en_example['entity_token_ids']).unsqueeze(0))
                entites_dataset_all_mask.append(torch.LongTensor(en_example['entity_token_mask']).unsqueeze(0))
                entites_dataset_all_type.append(torch.LongTensor(en_example['entity_token_type_ids']).unsqueeze(0))

        entites_dataset_all_token_tensor = torch.cat((entites_dataset_all_token),0)
        entites_dataset_all_mask_tensor = torch.cat((entites_dataset_all_mask),0)
        entites_dataset_all_type_tensor = torch.cat((entites_dataset_all_type),0)

        return {'entity_token_ids_hard':entites_dataset_all_token_tensor,
        'entity_token_type_ids_hard':entites_dataset_all_type_tensor,
        'entity_token_mask_hard':entites_dataset_all_mask_tensor,},entity_input_list_lable

    def neg_sampling(self,energy, batch_input,em_bank, batch_input_true_ids):
        energy.eval()

        #entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        #entity_tensor_all = torch.cat(entity_list, dim=0)
        x_input = batch_input[0]
        y_input = batch_input[1]
        x_input = util.move_to_cuda(x_input)
        

        true_tails = y_input['tail']
        true_head = y_input['head']
        true_r= y_input['relation']
        
        dic_hard_tails, hard_tails =self.hard_tails_em_naive(em_bank,y_input,batch_input_true_ids)

        
        

        return dic_hard_tails,hard_tails