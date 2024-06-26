# HaSa-CKGE
This is the script of the The web conference (WWW) 2024 🥣 paper of {[HaSa: Hardness and Structure-Aware Contrastive Knowledge
Graph Embedding](https://dl.acm.org/doi/abs/10.1145/3589334.3645564)}. 

![Diagram of paper](hasa_diag1.png)
![Diagram of paper](hasa_diag2.png)
## Requirements
Pytorch version 1.13 or above \
transformers version 4.26.1 or above\
networkx version 3.0\
pandas version 1.5.3 pr above\
## Hardware
We run experiments with 2 NV-V100-sxm2 GPUs.
## Training
Training the model to run the file

```console
honggen@hasa:~$ python main_WM_bert.py
```
## Link prediction

For the link prediction task, we evaluate the test data.

```console
honggen@hasa:~$ python evaluation.py
```

## Acknowledgements

Part of our code is inspired by 
https://github.com/intfloat/SimKGC, and https://github.com/chingyaoc/DCL/tree/master

