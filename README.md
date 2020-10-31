# OpenDialog

~~我们现在拥有了测试接口了，搜索微信公众号 `OpenDialog` 可以使用~~

OpenDialog建立在基于PyTorch的[transformers](https://github.com/huggingface/transformers)之上。
提供一系列transformer-based的**中文**开放域对话模型（闲聊对话），网罗已有的数据资源并持续不断的补充对应的中文对话系统的数据集，意图构建一个开源的中文闲聊对话平台。

__最新进展：__
* 2020.8.20, 完成[LCCC-GPT-Large](https://github.com/thu-coai/CDial-GPT)生成式Open-Domain预训练模型的接口，运行下面代码可以启动对应的服务

    ```bash
    ./run_flask lccc <gpu_id>
    ```
    
* 2020.10.26, 完成一批bi-encoder的检索式对话模型(bert-bi-encoder, polyencoder等)
    
* ...

## 使用教程

### 1. 项目结构和文件简述

OpenDialog核心文件和目录:

* `data`: 数据集，配置文件，词表，词向量，数据集处理脚本
* `models`: 对话模型
* `metrics`: 评价指标
* `multiview`: 多角度重排模型，针对获得对话候选回复进行重排序
* `ckpt`: 存放训练模型
* `rest`: 存放tensorboard日志和test阶段生成的结果文件
* `utils`: 存放工具函数
* `dataloader.py`: 数据集加载脚本
* `main.py`: 主运行文件
* `header.py`: 需要导入的package
* `eval.py`: 调用`metrics`中的评价指标的评估脚本，测试`rest`中生成文件的结果
* `run.sh`: 运行批处理脚本
* `run_flask.sh`: 调用模型，启动服务

### 2. 准备环境

1. 基础系统环境: `Linux/Ubuntu-16.04+`, `Python 3.6+`, `GPU (default 1080 Ti)`

2. 安装python依赖库

```bash
pip install -r requirements.txt
```

3. 安装 `ElasticSearch`

    基于检索的对话系统需要首先使用`elasticsearch`进行粗筛。同时为了实现粗筛检索阶段的中文分词，同时需要下载和安装[中文分词器](https://github.com/medcl/elasticsearch-analysis-ik)

4. 安装 `mongodb`

    启动服务之后，会使用`mongodb`存储会话历史和必要的数据

### 3. 准备数据

1. 数据集百度云链接: https://pan.baidu.com/s/1xJibJmOOCGIzmJVC6CZ39Q; 提取码: vmua
2. 将对应的数据文件存放在`data`目录下对应的子目录中，词向量文件`chinese_w2v.txt`和`english_w2v.bin`存放在`data`下即可。
3. 数据细节和预处理数据详见`data/README.md`。
4. 可用的数据集

### 5. 训练模型

* 训练模型支持多GPU并行，只需要`<gpu_ids>`指定多个gpu id即可，比如`0,1,2,3`
* `dataset`名称和`data`目录下的名称一致

| Model         | CMD                                              | Type       | Details | Refer | Pre-train Model |
|---------------|--------------------------------------------------|------------|---------| ----- | --------------- |
| bertretrieval | ./run.sh train \<dataset\> bertretrieval \<gpu_ids\> | retrieval  | 基于bert的精排模型(fine-tuning) | [Paper](https://arxiv.org/abs/1908.04812) | |
| gpt2          | ./run.sh train \<dataset\> gpt2 \<gpu_ids\>          | generative | GPT2生成式对话模型 | [Code](https://github.com/yangjianxin1/GPT2-chitchat) | |
| gpt2gan       | ./run.sh train \<dataset\> gpt2gan \<gpu_ids\>       | generative | GAN-based对话模型，生成式模型是GPT2，判别模型是bert二分类模型 | [Paper](https://arxiv.org/abs/1701.06547) | |

### 6. 实验结果

### 7. 启动flask服务

1. 启动flask服务
    ```
    ./run_flask.sh <model_name> <gpu_id>
    ```
    
2. 调用接口
    * 微信公众号
    * postman
