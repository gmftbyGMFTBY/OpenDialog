import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter
import numpy as np
from math import *
from queue import *
import jieba.analyse
import ipdb
import json
import re
import pickle
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import Counter, OrderedDict
from torch.nn.utils import clip_grad_norm_
import random
from elasticsearch import Elasticsearch, helpers
from .model_utils import *
from .base import *
from bert_serving.client import BertClient
from transformers.modeling_gpt2 import GPT2Model, GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertModel, BertForMultipleChoice
from transformers import AutoModelWithLMHead, AutoTokenizer, OpenAIGPTLMHeadModel
import transformers
import requests
# for PONE
from scipy.stats import pearsonr, spearmanr
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model

import sys
sys.path.append('..')
from metrics import *
from eval import *
