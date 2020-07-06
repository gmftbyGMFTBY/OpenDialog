1. resouce: https://github.com/CLUEbenchmark/CLUE; 使用wiki2019数据集进行语言模型训练
2. 大规模中文预训练语言模型的基础上进行fine-tuning
    * 使用大规模非平行对话语料进行预训练，之后再对话语料上进行fine-tuning是通用对话预训练语言模型的有潜力的方向。但是效果并不是非常的理想 [lipiji paper](https://arxiv.org/pdf/2003.04195)
    * 主要的原因在于fine-tune阶段仍然把对话看作是LM任务，但是context到response之间存在明显的语义转导的过程，单纯的语言模型无法衡量这个转导过程。
    * 为了更好的发挥预训练LM的效果，再fine-tune阶段不能再视其为LM任务，而是需要建模这种转导过程
    * This version contains the half of the corpus
