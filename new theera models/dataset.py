import torch
import tokenizer


class W2VData(torch.utils.data.Dataset):
  def __init__(self, corpus, window_size=2):
    self.tokenizer = tokenizer.Tokenizer(corpus)
    self.data = []
    self.create_data(window_size)

  def create_data(self, window_size):
    for sentence in self.tokenizer.corpus:
      tokens = self.tokenizer.encode(sentence)
      for i, target in enumerate(tokens):
        context = tokens[max(0, i - window_size):i] + tokens[i + 1:i + window_size + 1]
        if len(context) != 2 * window_size: continue
        self.data.append((context, target))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    context, target = self.data[idx]
    return torch.tensor(context), torch.tensor(target)


class LangData(torch.utils.data.Dataset):
  def __init__(self, raw_data):
    self.tknz = (tokenizer.Tokenizer()).load_vocab("./vocab.txt")
    self.data = self.create_data(raw_data)

  def create_data(self, data):
    lang_map = {lang: idx for idx, lang in enumerate(data.columns)}
    store = []
    for lang in data.columns:
      for sentence in data[lang]:
        store.append((sentence, lang_map[lang], lang))
    return store

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    sentence, target, lang = self.data[idx]
    return torch.tensor(self.tknz.encode(sentence)), torch.tensor(target), lang


if __name__ == '__main__':
  import pandas as pd
  data = pd.read_parquet('./Flores7Lang.parquet')
  w2v_long_format = data.melt(value_vars=data.columns)
  w2v_corpus = w2v_long_format['value'].tolist()
  w2v_ds = W2VData(w2v_corpus, 3)
  print("word2vec:corpus[0]", w2v_corpus[0])
  print("word2vec:ds[0]", w2v_ds[0])
  lang_ds = LangData(data)
  print("lang:ds[510]", lang_ds[510])