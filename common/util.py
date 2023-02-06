import torch
import re

def preprocess(text):
    """对文本的预处理

    Args:
        text (string): 输入的单词文本。

    Returns:
        tuple: 第一个值为语料库，
        第二个和第三个值都是字典，分别是用词获取id和用id获取词
    """
    word_to_id = {}
    id_to_word = {}

    words = re.findall(r'\b\w+\b|[^\w\s]+', text)
    words = [word.lower() for word in words]


    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = torch.tensor([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    '''生成共现矩阵

    :param corpus: 语料库（单词ID列表）
    :param vocab_size:词汇个数
    :param window_size:窗口大小（当窗口大小为1时，左右各1个单词为上下文）
    :return: 共现矩阵
    '''
    corpus_size = len(corpus)
    co_matrix = torch.zeros((vocab_size, vocab_size))

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    """计算余弦相似度

    Args:
        x (_type_): 第一个向量
        y (_type_): 第二个向量
        eps (_type_, optional): 用于防止“除数为0”的微小值. Defaults to 1e-8.

    Returns:
        _type_: 余弦相似度
    """

    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)