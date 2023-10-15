
import torch.nn as nn
import torch
import util
from random import sample





class NegativeSampleHard(object):
    
    def __init__(self,batch_size = 256, false_neg_size=4, hard_size=5, entites_dataset_all=None,eneities_labels=None,graph_stru=None,dic_triplet = None):
        super(NegativeSampleHard, self).__init__()

        self.negative_hard_size = hard_size
        self.false_negative_size = false_neg_size 
        self.entites_dataset_all = entites_dataset_all 
        self.eneities_labels = eneities_labels
        self.graph_stru = graph_stru
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
        #sort the entities based on similarity and return some hard negatives
        dic_hard_tails, hard_tails =self.hard_tails_em(em_bank,gx_input,y_input,batch_input_true_ids)

        false_score_list = []
        entity_input_list_lable = []
        entites_dataset_all_token = []
        entites_dataset_all_mask = []
        entites_dataset_all_type = []
        #find the false negative from Neighbor 
        for i in range(len(true_tails)):
            head = true_head[i]
            relation = true_r[i]
            hard_tail_i = hard_tails[i:i+self.negative_hard_size]
            false_neg_list1 = [hard_tail_i_each for hard_tail_i_each in hard_tail_i if hard_tail_i_each in self.graph_stru[head]['2']]
            num2 = self.false_negative_size- len(false_neg_list1)
            if num2>0:
                try:
                    false_neg_list2 = sample(self.graph_stru[head]['2'],num2)
                except:
                    other_true_entity= self.dic_triplet[head+'@'+relation]*num2
                    false_neg_list2 = sample(other_true_entity,num2)
                false_neg_list1.extend(false_neg_list2)
            false_neg_list = false_neg_list1[:self.false_negative_size]
            false_entity_list = []

            entity_input_list_lable.extend(false_neg_list)


            for en_name in false_neg_list:
                false_entity_list.append(em_bank[en_name].unsqueeze(0))

                en_example = self.entites_dataset_all[self.eneities_labels.index(en_name)]
                entites_dataset_all_token.append(torch.LongTensor(en_example['entity_token_ids']).unsqueeze(0))
                entites_dataset_all_mask.append(torch.LongTensor(en_example['entity_token_mask']).unsqueeze(0))
                entites_dataset_all_type.append(torch.LongTensor(en_example['entity_token_type_ids']).unsqueeze(0))


            #the false negatives and its probabiltiy
            false_entity_tensor = torch.cat(false_entity_list, dim=0)
            false_score_i =  torch.matmul(gx_input[i],torch.transpose(false_entity_tensor, 0, 1))
            false_score_list.append(false_score_i.unsqueeze(0))

        

        # batch data, hard data, and false data to build all tails
        #y_tails = y_input['tail']+y_input['head']+hard_tails+entity_input_list_lable
        y_tails = y_input['tail']+y_input['head']+hard_tails
        entity_list = [em_bank[en_name].unsqueeze(0) for en_name in y_tails]
        entity_tensor_all = torch.cat(entity_list, dim=0)



        # return their socre and assign it as zero if we know it is false tails
        p_score =  torch.matmul(gx_input,torch.transpose(entity_tensor_all, 0, 1))
        #p_score_exp = torch.exp(p_score)#1024*1
        p_score_exp = nn.functional.normalize(p_score,dim = 1)
        #print(p_score_exp)
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
        #false negative **data
        entites_dataset_all_token_tensor = torch.cat((entites_dataset_all_token),0)
        entites_dataset_all_mask_tensor = torch.cat((entites_dataset_all_mask),0)
        entites_dataset_all_type_tensor = torch.cat((entites_dataset_all_type),0)

        false_score_tensor = torch.cat(false_score_list, dim=0)
        false_score_tensor = nn.functional.normalize(false_score_tensor,dim = 1)

        f_prob = nn.functional.softmax(false_score_tensor,dim = 1)
        #print('f_prob',f_prob.shape)256*3
 
        prob = nn.functional.softmax(p_score_exp,dim = 1)
        #print('prob',prob.shape)256*1280
        for l in range(self.batch_size):
            prob[l][l] = 1


        #return prob of tails, prob of false negative tails, false ngative data, false negative label, hard negative data, hard negative label
        return prob,f_prob,{'entity_token_ids':entites_dataset_all_token_tensor,
        'entity_token_type_ids':entites_dataset_all_type_tensor,
        'entity_token_mask':entites_dataset_all_mask_tensor,},entity_input_list_lable,dic_hard_tails,hard_tails


class NegativeSampleHard_yg(object):
    
    def __init__(self,batch_size = 256, false_neg_size=4, hard_size=5, entites_dataset_all=None,eneities_labels=None,dic_triplet = None):
        super(NegativeSampleHard_yg, self).__init__()

        self.negative_hard_size = hard_size
        self.false_negative_size = false_neg_size 
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




    def neg_sampling(self,energy, batch_input,em_bank, batch_input_true_ids,graph_stru):
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

        false_score_list = []
        entity_input_list_lable = []
        entites_dataset_all_token = []
        entites_dataset_all_mask = []
        entites_dataset_all_type = []
        for i in range(len(true_tails)):
            head = true_head[i]
            relation = true_r[i]
            hard_tail_i = hard_tails[i:i+self.negative_hard_size]
            false_neg_list1 = [hard_tail_i_each for hard_tail_i_each in hard_tail_i if hard_tail_i_each in graph_stru[head]['2']]
            num2 = self.false_negative_size- len(false_neg_list1)
            if num2>0:
                try:

                    false_neg_list2 = sample(graph_stru[head]['2'],num2)
                except:
                    other_true_entity= self.dic_triplet[head+'@'+relation]*num2
                    false_neg_list2 = sample(other_true_entity,num2)
                false_neg_list1.extend(false_neg_list2)
            false_neg_list = false_neg_list1[:self.false_negative_size]
            false_entity_list = []

            entity_input_list_lable.extend(false_neg_list)


            for en_name in false_neg_list:
                false_entity_list.append(em_bank[en_name].unsqueeze(0))

                en_example = self.entites_dataset_all[self.eneities_labels.index(en_name)]
                entites_dataset_all_token.append(torch.LongTensor(en_example['entity_token_ids']).unsqueeze(0))
                entites_dataset_all_mask.append(torch.LongTensor(en_example['entity_token_mask']).unsqueeze(0))
                entites_dataset_all_type.append(torch.LongTensor(en_example['entity_token_type_ids']).unsqueeze(0))



            false_entity_tensor = torch.cat(false_entity_list, dim=0)
            false_score_i =  torch.matmul(gx_input[i],torch.transpose(false_entity_tensor, 0, 1))
            false_score_list.append(false_score_i.unsqueeze(0))

        


        y_tails = y_input['tail']+y_input['head']+hard_tails+entity_input_list_lable
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
        #print(p_score_exp)
        entites_dataset_all_token_tensor = torch.cat((entites_dataset_all_token),0)
        entites_dataset_all_mask_tensor = torch.cat((entites_dataset_all_mask),0)
        entites_dataset_all_type_tensor = torch.cat((entites_dataset_all_type),0)

        false_score_tensor = torch.cat(false_score_list, dim=0)
        false_score_tensor = nn.functional.normalize(false_score_tensor,dim = 1)

        f_prob = nn.functional.softmax(false_score_tensor,dim = 1)
        #print(f_prob)
 
        prob = nn.functional.softmax(p_score_exp,dim = 1)
        #print(prob)
        for l in range(self.batch_size):
            prob[l][l] = 1



        return prob,f_prob,{'entity_token_ids':entites_dataset_all_token_tensor,
        'entity_token_type_ids':entites_dataset_all_type_tensor,
        'entity_token_mask':entites_dataset_all_mask_tensor,},entity_input_list_lable,dic_hard_tails,hard_tails






class NegativeSampleHard_check(object):
    
    def __init__(self,batch_size = 256, false_neg_size=4, hard_size=5, entites_dataset_all=None,eneities_labels=None,graph_stru=None,dic_triplet = None):
        super(NegativeSampleHard_check, self).__init__()

        self.negative_hard_size = hard_size
        self.false_negative_size = false_neg_size 
        self.entites_dataset_all = entites_dataset_all 
        self.eneities_labels = eneities_labels
        self.graph_stru = graph_stru
        self.batch_size = batch_size
        self.dic_triplet = dic_triplet
        
        
    def neg_random_check(self,em_bank,gx_input,y_input,batch_input_true_ids,dic_triplet_rest):
        entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        entity_tensor_all = torch.cat(entity_list, dim=0)

        p_score =  torch.matmul(gx_input,torch.transpose(entity_tensor_all, 0, 1))
        #p_score_exp = torch.exp(p_score)#1024*1
        true_tails = y_input['tail']
        true_head = y_input['head']
        true_relation = y_input['relation']

        #sorted_score, sorted_indices = torch.sort(p_score_exp, dim=-1, descending=True)
        
        sorted_score = p_score
        for i in range(self.batch_size):
            #k = eneities_labels.index(true_tails[i])
            sorted_score[i][i] = -10
            other_true_ids= batch_input_true_ids[i]
            for j in other_true_ids:
                false_tail = self.eneities_labels[j]
                try:
                    #k_list = [k for k in range(len(a)) if a[k] == 'b']
                    k = true_tails.index(false_tail)
                    sorted_score[i][k] = -10
                except:
                    pass

        entity_input_list_lable_all = []
        index_list_all = []
        for i in range(self.batch_size):
            head_label = true_head[i]
            relation_label = true_relation[i]
            try:
                true_tails = dic_triplet_rest[head_label+'@'+relation_label]
            except:
                true_tails = []
            index_list = sample(list(range(len(self.eneities_labels))),self.negative_hard_size)
            index_list_all.append(index_list)


        entity_input_list_lable = {}
        entity_input_list_lable_false = {}
        entity_input_list_score = {}
        entity_input_list_score_false = {}
        for i in range(self.batch_size):
            head_label = true_head[i]
            relation_label = true_relation[i]
            entity_input_list_lable_each = []
            entity_input_list_lable_each_false = []
            entity_input_list_score_each = []
            entity_input_list_score_each_false = []
            try:
                true_tails = dic_triplet_rest[head_label+'@'+relation_label]
            except:
                true_tails = []
            for j in range(self.negative_hard_size):
                k = index_list_all[i][j]
                score = sorted_score[i][k].item()
                if score >-10:

                    ent_label =self.eneities_labels[k]
                    if ent_label in true_tails:
                        entity_input_list_lable_each_false.append(ent_label)
                        entity_input_list_score_each_false.append(score)
                    else:
                        entity_input_list_lable_each.append(ent_label)
                        entity_input_list_score_each.append(score)

            entity_input_list_lable[head_label+'@'+relation_label] = entity_input_list_lable_each
            entity_input_list_lable_false[head_label+'@'+relation_label] = entity_input_list_lable_each_false
            entity_input_list_score[head_label+'@'+relation_label] = entity_input_list_score_each
            entity_input_list_score_false[head_label+'@'+relation_label] = entity_input_list_score_each_false  



        

        return {'true_tail_lable': entity_input_list_lable,
            'true_tail_score': entity_input_list_score,
            'false_tail_lable': entity_input_list_lable_false,
            'false_tail_score': entity_input_list_score_false}
        
    def neg_hard_check(self,em_bank,gx_input,y_input,batch_input_true_ids,dic_triplet_rest):
        entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        entity_tensor_all = torch.cat(entity_list, dim=0)

        p_score =  torch.matmul(gx_input,torch.transpose(entity_tensor_all, 0, 1))
        p_score_exp = p_score#1024*1
        true_tails = y_input['tail']
        true_head = y_input['head']
        true_relation = y_input['relation']
        for i in range(len(true_tails)):
            k = self.eneities_labels.index(true_tails[i])
            p_score_exp[i][k] = -10
            other_true_ids= batch_input_true_ids[i]
            for j in other_true_ids:
                #k = other_true_ids[j]
                p_score_exp[i][j] = -10




        sorted_score, sorted_indices = torch.sort(p_score_exp, dim=-1, descending=True)



        entity_input_list_lable = {}
        entity_input_list_lable_false = {}
        entity_input_list_score = {}
        entity_input_list_score_false = {}
        for i in range(self.batch_size):
            head_label = true_head[i]
            relation_label = true_relation[i]
            entity_input_list_lable_each = []
            entity_input_list_lable_each_false = []
            entity_input_list_score_each = []
            entity_input_list_score_each_false = []
            try:
                true_tails = dic_triplet_rest[head_label+'@'+relation_label]
            except:
                true_tails = []
            for j in range(2*self.negative_hard_size):
                score = sorted_score[i][j].item()
                k = sorted_indices[i][j].item()
                ent_label = self.eneities_labels[k]
                if ent_label in true_tails:
                    entity_input_list_lable_each_false.append(ent_label)
                    entity_input_list_score_each_false.append(score)
                else:
                    entity_input_list_lable_each.append(ent_label)
                    entity_input_list_score_each.append(score)
            
            entity_input_list_lable[head_label+'@'+relation_label] = entity_input_list_lable_each
            entity_input_list_lable_false[head_label+'@'+relation_label] = entity_input_list_lable_each_false
            entity_input_list_score[head_label+'@'+relation_label] = entity_input_list_score_each
            entity_input_list_score_false[head_label+'@'+relation_label] = entity_input_list_score_each_false  



        

        return {'true_tail_lable': entity_input_list_lable,
            'true_tail_score': entity_input_list_score,
            'false_tail_lable': entity_input_list_lable_false,
            'false_tail_score': entity_input_list_score_false}



    def neg_sampling_check(self,energy, batch_input,em_bank, batch_input_true_ids, dic_triplet_rest):
        energy.eval()

        #entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        #entity_tensor_all = torch.cat(entity_list, dim=0)
        x_input = batch_input[0]
        y_input = batch_input[1]
        x_input = util.move_to_cuda(x_input)
        gx_input,_,_ = energy.module.predict_mapping_embedding(**x_input)


        true_tails = y_input['tail']
        true_head = y_input['head']
        true_r = y_input['relation']



        dic_hard=self.neg_hard_check(em_bank,gx_input,y_input,batch_input_true_ids,dic_triplet_rest)
        #dic_hard=self.neg_random_check(em_bank,gx_input,y_input,batch_input_true_ids,dic_triplet_rest)

        #y_tails = y_input['tail']+y_input['head']+hard_tails+entity_input_list_lable
        y_tails = y_input['tail']+y_input['head']
        entity_list = [em_bank[en_name].unsqueeze(0) for en_name in y_tails]
        entity_tensor_all = torch.cat(entity_list, dim=0)




        p_score =  torch.matmul(gx_input,torch.transpose(entity_tensor_all, 0, 1))
        
        #p_score_exp = torch.exp(p_score)#1024*1
        p_score_exp = p_score
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
                
        sorted_indices = [list(range(2*self.batch_size)) for i in range(self.batch_size)]
        sorted_score = p_score_exp

        entity_input_list_lable = {}
        entity_input_list_lable_false = {}
        entity_input_list_score = {}
        entity_input_list_score_false = {}
    
        for i in range(self.batch_size):
            head_label = true_head[i]
            relation_label = true_r[i]
            entity_input_list_lable_each = []
            entity_input_list_lable_each_false = []
            entity_input_list_score_each = []
            entity_input_list_score_each_false = []
            try:
                true_tails = dic_triplet_rest[head_label+'@'+relation_label]
            except:
                true_tails = []
            for j in range(2*self.batch_size):
                k = sorted_indices[i][j]
                score = sorted_score[i][j].item()
                if score>-10:

                    ent_label = y_tails[k]
                    if ent_label in true_tails:
                        entity_input_list_lable_each_false.append(ent_label)
                        entity_input_list_score_each_false.append(score)
                    else:
                        entity_input_list_lable_each.append(ent_label)
                        entity_input_list_score_each.append(score)
            
            entity_input_list_lable[head_label+'@'+relation_label] = entity_input_list_lable_each
            entity_input_list_lable_false[head_label+'@'+relation_label] = entity_input_list_lable_each_false
            entity_input_list_score[head_label+'@'+relation_label] = entity_input_list_score_each
            entity_input_list_score_false[head_label+'@'+relation_label] = entity_input_list_score_each_false  




        return {'true_tail_lable': entity_input_list_lable,
            'true_tail_score': entity_input_list_score,
            'false_tail_lable': entity_input_list_lable_false,
            'false_tail_score': entity_input_list_score_false},dic_hard
    
