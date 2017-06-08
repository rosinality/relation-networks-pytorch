# relation-networks-pytorch
Relation Networks (https://arxiv.org/abs/1706.01427) for CLEVR implemented in PyTorch

To train:

1. Download and extract CLEVR v1.0 dataset from http://cs.stanford.edu/people/jcjohns/clevr/
2. Preprocessing question data
```
python preprocess.py [CLEVR directory]
```
3. Run train.py
```
python train.py [CLEVR directory]
```