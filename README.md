# HaSa-CKGE
This is the script of the The web conference (WWW) 2024 🐫 paper of HaSa: Hardness and Structure-Aware Contrastive Knowledge
Graph Embedding {https://arxiv.org/pdf/2305.10563v2.pdf}. 

## Requirements
Pytorch version 1.13 or above \
transformers version 4.26.1 or above\
networkx version 3.0\
pandas version 1.5.3 pr above\
## Hardware
We run experiments with 2 NV-V100-sxm2 GPUs.
## Training
Training the model to run the file\
```
python main_WN.py
```
## Link prediction

For the link prediction task, we evaluate the test data.
```
python evaluation.py
```

## Acknowledgements

Part of our code is inspired by 
https://github.com/intfloat/SimKGC, and https://github.com/chingyaoc/DCL/tree/master

