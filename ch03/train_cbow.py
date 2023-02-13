import sys
sys.path.append('.')
from simple_cbow import SimpleCBOW
from common.trainer import Trainer
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
# target = convert_one_hot(target, vocab_size)
# contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size=vocab_size, hidden_size=hidden_size)
trainer = Trainer(model)
trainer.fit(contexts, target, max_epoch, batch_size, eval_interval=10)
# trainer.plot()

word_vecs = model.embedding.weight.data
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])