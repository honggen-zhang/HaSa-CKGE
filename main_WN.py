import torch
import torch.backends.cudnn as cudnn
import logging
from train import train

ngpus_per_node = torch.cuda.device_count()
logging.info('# gpus: %s',ngpus_per_node)
'''
For the HaSa method, using the 'HaSa'. 
For the Hard InfoNCE, using 'HaSa_Hard_Bias'. 
For the simple InfoNCE, using 'HaSa_wohard_Bias'
Using plus = True will boost HaSa as HaSa+
'''

batch_size = 256
learn_rate = 2e-5
num_false_neg = 3
num_hard_neg = 3
data_file_path = '/data/benchmarks/'
output_file_path = '/models/WN/'

Hasa_model = 'HaSa'
#Hasa_model = 'HaSa_Hard_Bias'
#Hasa_model = 'HaSa_wohard_Bias'
pretrain_model = 'sentence-transformers/all-mpnet-base-v2'#using the sentence-bert as backbone
input_dim = 768
em_dim = 500
epoch_step = 10
tau = 2e-5
data_name = 'WN18RR/' 
Resuming = False
#loading_model = '/bestmodel.pth.tar'
loading_model = None
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
      schedule_with_warmup=True,
      plus = False)



