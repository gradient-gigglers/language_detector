import string


class Tokenizer:
  def __init__(self, corpus=None, freq_threshold=1):
    self.corpus = corpus
    self.freq_threshold = freq_threshold
    self.freq_dist = self.build_freq_dist() if corpus else {}
    self.vocab = self.build_vocab() if corpus else {}
    self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

  def build_freq_dist(self):
    freq_dist = {}
    for sentence in self.corpus:
      for word in sentence.split():
        word = self.clean_word(word)
        # Check if the word is not empty after cleaning
        if word: freq_dist[word] = freq_dist.get(word, 0) + 1
    return freq_dist

  def build_vocab(self):
    tokens = [word.lower() for sentence in self.corpus for word in sentence.split()]
    tokens = [self.clean_word(word) for word in tokens if word]  # Clean and filter out empty words
    vocab = list({word for word in tokens if self.freq_dist.get(word, 0) > self.freq_threshold})
    return vocab

  def save_vocab(self, path):
    with open(path, 'w') as f:
      for word in self.vocab: f.write(word + '\n')
    return self

  def load_vocab(self, path):
    with open(path, 'r') as f: self.vocab = [line.strip() for line in f.readlines()]
    self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
    return self

  def encode(self, sentence):
    words = sentence.split()
    words = [self.clean_word(word) for word in words if word]  # Clean and filter out empty words
    return [self.word2idx[word] for word in words if word in self.word2idx]

  def decode(self, indices):
    return ' '.join(self.idx2word[idx] for idx in indices if idx in self.idx2word)

  @staticmethod
  def clean_word(word):
    word = word.lower()
    word = ''.join(char for char in word if char not in string.punctuation)
    word = ''.join(char for char in word if not char.isdigit())
    return word.strip()


if __name__ == '__main__':
  import pandas as pd
  data = pd.read_parquet('./Flores7Lang.parquet')
  long_format = data.melt(value_vars=data.columns)
  corpus = long_format['value'].tolist()
  tknz = Tokenizer(corpus)
  tknz.save_vocab('./vocab.txt')
  tknz.load_vocab('./vocab.txt')
  print(len(tknz.vocab))
  print(tknz.vocab[90:100])