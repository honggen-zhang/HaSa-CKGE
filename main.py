import torch
import torch.backends.cudnn as cudnn
#import logging
from train import train
from train_bert import train_bert
ngpus_per_node = torch.cuda.device_count()
print('# gpus:',ngpus_per_node)

batch_size = 256
learn_rate = 2e-5
#batch_size = 64
num_false_neg = 3
num_hard_neg = 3
data_file_path = 'hasa/data/benchmarks/'
output_file_path = 'dynamicgraph_lts/Honggen/HaSa/modelWWW/FB_plus/'
Hasa_model = 'HaSa'
#Hasa_model = 'HaSa_Hard_Bias'
#Hasa_model = 'HaSa_wohard_Bias'
pretrain_model = 'sentence-transformers/all-mpnet-base-v2'
#pretrain_model = 'bert-base-uncased'
input_dim = 768
em_dim = 500
epoch_step = 5
tau = 1e-4
data_name = 'FB15K237_L/' 
Resuming = True
loading_model = 'dynamicgraph_lts/Honggen/HaSa/modelWWW/FB_plus/bestmodel_1e4_8.pth.tar'
#loading_model = None
train(batch_size = batch_size,
           learn_rate = learn_rate,
      num_false_neg = num_false_neg,
      num_hard_neg = num_hard_neg,
      data_file_path =data_file_path,
      output_file_path = output_file_path,
      Hasa_model = Hasa_model,
      pretrain_model = pretrain_model,
      input_dim = input_dim,
      em_dim = em_dim,
      epoch_step = epoch_step,
      tau = tau,
      data_name =data_name,
      Resuming=Resuming,
      loading_model=loading_model,
      schedule_with_warmup=False,
      plus = True)

