import torch
import torch.backends.cudnn as cudnn
#import logging
from eva import eva
import os

ngpus_per_node = torch.cuda.device_count()
print('# gpus:',ngpus_per_node)

batch_size = 256
#batch_size = 128
num_false_neg = 3
num_hard_neg = 3
data_file_path = 'hasa/data/benchmarks/'
Hasa_model = 'HaSa'
#Hasa_model = 'HaSa_Hard_Bias'
#Hasa_model = 'HaSa_wohard_Bias'
pretrain_model = 'sentence-transformers/all-mpnet-base-v2'
#pretrain_model = 'bert-base-uncased'
input_dim = 768
em_dim = 500
#em_dim = 100
data_name = 'FB15K237_L/' 
#data_name = 'WN18RR/' 

#os_file = 'nfs_scratch/models_raw/FB/'
os_file = 'dynamicgraph_lts/Honggen/HaSa/modelWWW/FB_plus/'
filelist_folder = os.listdir(os_file)
filelist_folder.sort() 
i=1
for model_file_name in filelist_folder[:1]:
    model_file = os_file+'bestmodel_1e4_'+str(i+11)+'.pth.tar'
    #model_file = os_file+'bestmodel_1e4_27.pth.tar'
    i = i+1
    #model_file = os_file+model_file_name
    print(model_file)

    eva(batch_size = batch_size,
          num_false_neg = num_false_neg,
          num_hard_neg = num_hard_neg,
          data_file_path =data_file_path,
          Hasa_model = Hasa_model,
          pretrain_model = pretrain_model,
          input_dim = input_dim,
          em_dim = em_dim,
          data_name =data_name,
          model_file=model_file,)


