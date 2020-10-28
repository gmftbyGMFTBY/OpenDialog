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
| bert bi-encoder ocn (random) | 0.843 | 0.93 | 0.985 | 0.9047 |
| bert bi-encoder ocn (decay ratio=1.0) | 0.835 | 0.929 | 0.987 | 0.9003 |
| bert bi-encoder ocn (decay ratio=0.9) | 0.83 | 0.93 | 0.986 | 0.8979 |
| bert bi-encoder ocn (decay ratio=0.8) | 0.831 | 0.93 | 0.985 | 0.8983 |
| bert bi-encoder ocn (decay ratio=0.7) | 0.839 | 0.933 | 0.988 | 0.9031 |
| bert bi-encoder ocn (decay ratio=0.6) | 0.842 | 0.933 | 0.987 | 0.9047 |
| bert bi-encoder ocn (decay ratio=0.5) | 0.844 | 0.935 | 0.987 | 0.906 |
| bert bi-encoder ocn (decay ratio=0.4) | 0.85 | 0.94 | 0.988 | 0.9101 |
| bert bi-encoder ocn (decay ratio=0.3) | 0.852 | 0.94 | 0.987 | 0.9115 |
| bert bi-encoder ocn (decay ratio=0.2) | 0.853 | 0.94 | 0.988 | 0.912 |
| bert bi-encoder ocn (decay ratio=0.1) | 0.857 | 0.944 | 0.99 | 0.915 |
| bert bi-encoder ocn (decay ratio=0.0) | 0.859 | 0.944 | 0.99 | 0.9161 |
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