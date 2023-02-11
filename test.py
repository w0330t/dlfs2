
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pickle
import re

from common.util import preprocess, create_co_matrix

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

def create_contexts_target(corpus, window_size=1):
    '''生成上下文和目标词

    :param corpus: 语料库（单词ID列表）
    :param window_size: 窗口大小（当窗口大小为1时，左右各1个单词为上下文）
    :return:
    '''
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t].item())
        contexts.append(cs)

    return torch.tensor(contexts), target

contexts, target = create_contexts_target(corpus, window_size=1)




