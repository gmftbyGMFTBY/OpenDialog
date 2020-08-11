import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from models.bert_retrieval import BERTRetrieval
from models.bert_retrieval_multi import BERTMULTIVIEW
from models.gpt2 import GPT2
from models.base import RetrievalBaseAgent, BaseAgent
from models.bert_mc import BERTMCFusion
import fasttext.FastText as ff
import argparse
import numpy as np
import ipdb
import pprint
from tqdm import tqdm
from math import *
import pickle
import os
import jieba
import thulac
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
