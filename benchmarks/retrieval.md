## E-Commerce Dataset

### 1. Hyperparameters:
* epoch: 10
* max sequence length: 256
* negative samples: 32
* apex: True
* distributed: True
* batch size: 32
* seed: 50

### 2. experiment results


| Models           | R1@10 | R2@10 | R5@10 | MRR    |
|------------------|-------|-------|-------|------: |
| bert bi-encoder  | 0.846 | 0.93  | 0.986 | 0.9064 |
| bert bi-encoder ocn |       |      |       |       |
| bert polyencoder |       |       |       |       |
| bert cross-attention |      |       |       |       |


## Douban Multi-turn Conversation

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