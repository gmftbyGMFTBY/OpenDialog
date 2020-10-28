## E-Commerce Dataset

### 1. Hyperparameters:
* epoch: 10
* max sequence length: 256
* negative samples: 32
* warm up step: 8000
* apex: True
* distributed: True
* batch size: 32
* seed: 50
* decay ratio: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
* transformer parameters:
    * nhead: 6
    * dropout: 0.1
    * dim feedforward: 512
    * num encoder layer: 2

### 2. experiment results


| Models           | R1@10 | R2@10 | R5@10 | MRR    |
|------------------|-------|-------|-------|------: |
| bert bi-encoder  | 0.846 | 0.93  | 0.986 | 0.9064 |
| bert bi-encoder ocn | 0.849 | 0.934 | 0.985 | 0.9086 |
| bert polyencoder |       |       |       |       |
| bert cross-attention |      |       |       |       |


## Douban Multi-turn Conversation Dataset

### 1. Hyperparameters:
* epoch: 10
* max sequence length: 256
* negative samples: 16
* warm up step: 16000
* apex: True
* distributed: True
* batch size: 32
* seed: 50
* decay ratio: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
* transformer parameters:
    * nhead: 6
    * dropout: 0.1
    * dim feedforward: 512
    * num encoder layer: 2

### 2. experiment results

| Models           | R1@10 | R2@10 | R5@10 | MRR   |
|------------------|-------|-------|-------|------:|
| bert bi-encoder  |  0.2762     |    0.4751   |  0.8177     |  0.4931     |
| bert bi-encoder ocn |       |      |       |       |
| bert polyencoder |       |       |       |       |
| bert cross-attention |      |       |       |       |