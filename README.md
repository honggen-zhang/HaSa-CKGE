# HaSa-CKGE
This is the script of paper of Investigating the Effect of Hard Negative Sample Distribution on Contrastive Knowledge Graph Embedding. We call our algorithm Hardness and Structure-aware (HaSa) contrastive KGE.

## Requirements
Pytorch version 1.13 or above \
transformers version 4.26.1 or above\
networkx version 3.0\
pandas version 1.5.3 pr above\
## Hardware
We run experiments with 2 NV-V100-sxm2 GPUs.
## Training
Traning the model to run the file\
```
python example/main_WN.py
```
## Link prediction

For the link preduction task, we evaluate on the test data.
```
python example/evaluation.py
```

## Acknowledgements

Part of our code is inspired by 
https://github.com/intfloat/SimKGC, and https://github.com/chingyaoc/DCL/tree/master

