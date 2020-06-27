## Multi-view Metric for Open-domain Dialog Evaluation
单一的分数评价开放域对话系统已经不足了，需要多种评价方式

1. coherence: bert 检索模型可以提供一致性信息
2. fluency: 主要针对生成式对话系统，生成的句子是否流畅，语义清晰
3. topic: topic 文本分类模型对主题进行分类判断，主要针对SMP-MCC 2020比赛
4. diversity: 多样性
    * Distinct: 计算Micro-distinct和Macro-distinct分数
    * NIDF and NTF: 归一化IDF和TF分数
    * Length: 长度计算，惩罚短文本
    * Repetition Penalty: 重复性惩罚，对重复生成的token进行计算并压低对应分数 
5. MMI: 互信息最大化，来自论文DialoGPT
