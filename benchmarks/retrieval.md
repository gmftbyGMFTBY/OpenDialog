## E-Commerce Dataset

### 1. Hyperparameters:
* epoch: 10
* max sequence length: 256
* negative samples: 32 (only for bi-encoder, cross-attention is 1)
* apex: True
* distributed: True
* batch size: 32
* seed: 50
* decay ratio: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

### 2. experiment results


| Models           | R1@10 | R2@10 | R5@10 | MRR    |
|------------------|-------|-------|-------|------: |
| bert bi-encoder  | 0.846 | 0.93  | 0.986 | 0.9064 |
| bert bi-encoder ocn (decay ratio=1.0) | 0.84 | 0.928 | 0.986 | 0.9033 |
| bert bi-encoder ocn (decay ratio=0.9) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.8) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.7) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.6) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.5) | 0.861 | 0.946 | 0.99 | 0.9181 |
| bert bi-encoder ocn (decay ratio=0.4) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.3) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.2) |  |  |  |  |
| bert bi-encoder ocn (decay ratio=0.1) |  |  |  |  |
| bert polyencoder |       |       |       |       |
| bert cross-attention |      |       |       |       |


## Douban Multi-turn Conversation Dataset

### 1. Hyperparameters:
* epoch: 10
* max sequence length: 256
* negative samples: 32
* apex: True
* distributed: True
* batch size: 32
* seed: 50

### 2. experiment results

| Models           | R1@10 | R2@10 | R5@10 | MRR   |
|------------------|-------|-------|-------|------:|
| bert bi-encoder  |       |       |       |       |
| bert bi-encoder ocn |       |      |       |       |
| bert polyencoder |       |       |       |       |
| bert cross-attention |      |       |       |       |