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
* transformer parameters:
    * nhead: 6    ->    8
    * dropout: 0.1
    * dim feedforward: 512
    * num encoder layer: 2    ->    4
* poly m: 16

### 2. experiment results

#### 2.1 10 Candidates

| Models           | R1@10 | R2@10 | R5@10 | MRR    |
|------------------|-------|-------|-------|------: |
| bert bi-encoder  | 0.846 | 0.93  | 0.986 | 0.9064 |
| bert bi-encoder ocn | 0.849 | 0.934 | 0.985 | 0.9086 |
| bert polyencoder |  0.84  |  0.932   |   0.987  |    0.9036  |

#### 2.2 50 Candidates

| Models           | R1@50 | R2@50 | R5@50 | R10@50 | MRR    |
|------------------|-------|-------|-------|-------|------: |
| bert bi-encoder  | 0.644 | 0.762  | 0.856 | 0.912 | 0.7413 | 
| bert bi-encoder ocn | 0.671 | 0.762  | 0.861 | 0.921 |  0.7565 |
| bert polyencoder | 0.671 | 0.773  | 0.875 | 0.925 | 0.7615 |

#### 2.3 100 Candidates

| Models           | R1@100 | R2@100 | R5@100 | R10@100 | MRR    |
|------------------|-------|-------|-------|-------|------: |
| bert bi-encoder  | 0.603 | 0.7   | 0.809 | 0.872 | 0.6961 |
| bert bi-encoder ocn | 0.638 | 0.732 | 0.814 | 0.872 | 0.7273 | 
| bert polyencoder |  0.628  |   0.73  |  0.831  |  0.883  | 0.7203 |

#### 2.4 150 Candidates

| Models           | R1@150 | R2@150 | R5@150 | R10@150 | MRR    |
|------------------|-------|-------|-------|-------|------: |
| bert bi-encoder  | 0.569 | 0.676 | 0.773 | 0.842 | 0.6657 |
| bert bi-encoder ocn | 0.604 | 0.696  | 0.798 | 0.855 | 0.692 |
| bert polyencoder | 0.603  |  0.718 | 0.809 | 0.862 | 0.6977 |

#### 2.5 200 Candidates

| Models           | R1@200 | R2@200 | R5@200 | R10@200 | MRR    |
|------------------|-------|-------|-------|-------|------: |
| bert bi-encoder  | 0.551 | 0.651 | 0.758 | 0.831 | 0.6473 |
| bert bi-encoder ocn | 0.588 |  0.684 | 0.782 | 0.841 | 0.6768 |
| bert polyencoder | 0.584 | 0.695  | 0.788 | 0.846 | 0.6787 |

#### 2.6 250 Candidates

| Models           | R1@250 | R2@250 | R5@250 | R10@250 | MRR    |
|------------------|-------|-------|-------|-------|------: |
| bert bi-encoder  | 0.532 | 0.643 | 0.734 | 0.806 | 0.6298 |
| bert bi-encoder ocn | 0.573 | 0.66  |  0.758 | 0.825 | 0.6594 |
| bert polyencoder | 0.574 | 0.677  | 0.769 | 0.836 | 0.6664 |

#### 2.7 300 Candidates

| Models           | R1@300 | R2@300 | R5@300 | R10@300 | MRR    |
|------------------|-------|-------|-------|-------|------: |
| bert bi-encoder  | 0.529 | 0.623 | 0.723 | 0.796 | 0.6217 |
| bert bi-encoder ocn | 0.555 | 0.653  | 0.747 | 0.814 | 0.6468 |
| bert polyencoder | 0.56 |  0.659 | 0.765 | 0.825 | 0.6527 |


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
* transformer parameters:
    * nhead: 6   ->   8
    * dropout: 0.1
    * dim feedforward: 512
    * num encoder layer: 2    ->    4
* poly m: 16

### 2. experiment results

| Models           | R1@10 | R2@10 | R5@10 | MRR   |
|------------------|-------|-------|-------|------:|
| bert bi-encoder  |  0.2762     |    0.4751   |  0.8177     |  0.4931     |
| bert bi-encoder ocn |   0.3039    |   0.4613   |   0.8122    |   0.5041    |
| bert polyencoder (m=16) |  0.2873     |   0.4586    |  0.8066     |   0.4952    |
