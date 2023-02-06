import torch
import re

def preprocess(text):
    word_to_id = {}
    id_to_word = {}

    words = re.findall(r'\b\w+\b', text)
    words = [word.lower() for word in words]


    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = torch.tensor([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word