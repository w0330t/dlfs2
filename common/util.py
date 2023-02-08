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


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''相似单词的查找

    :param query: 查询词
    :param word_to_id: 从单词到单词ID的字典
    :param id_to_word: 从单词ID到单词的字典
    :param word_matrix: 汇总了单词向量的矩阵，假定保存了与各行对应的单词向量
    :param top: 显示到前几位
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    # 1.取出查询词的单词向量。
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = torch.zeros(vocab_size)
    # 2.分别求得查询词的单词向量和其他所有单词向量的余弦相似度。
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 3.基于余弦相似度的结果，按降序显示它们的值。
    count = 0
    for i in (-1 * similarity).argsort():
        i = i.item()
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps = 1e-8):
    '''生成PPMI（正的点互信息）

    :param C: 共现矩阵
    :param verbose: 是否输出进展情况
    :return:
    '''
    M = torch.zeros_like(C)
    N = torch.sum(C)
    S = torch.sum(C, dim=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = torch.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M