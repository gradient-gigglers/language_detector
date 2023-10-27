import torch


class CBOW(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CBOW, self).__init__()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.linear = torch.nn.Linear(embedding_dim, vocab_size)

  def forward(self, inputs):
    embeds = torch.sum(self.embeddings(inputs), dim=1)
    out = self.linear(embeds)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs


class SkipGram(torch.nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(SkipGram, self).__init__()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.linear = torch.nn.Linear(embedding_dim, vocab_size)

  def forward(self, target_word):
    embeds = self.embeddings(target_word)
    out = self.linear(embeds)
    log_probs = torch.nn.functional.log_softmax(out, dim=1)
    return log_probs


class Language(torch.nn.Module):
  def __init__(self, embedding_weights, num_classes=7):
    super(Language, self).__init__()
    vocab_size, embedding_dim = embedding_weights.size()
    self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
    self.embeddings.load_state_dict({'weight': embedding_weights})
    # self.embeddings.requires_grad = False  # Optional: Freeze embeddings
    self.linear = torch.nn.Linear(embedding_dim, num_classes)

  def forward(self, inputs):
    embeds = self.embeddings(inputs)
    # Average pooling along the sequence length
    pooled = torch.mean(embeds, dim=1)
    output = self.linear(pooled)
    return output


if __name__ == '__main__':
  cbow = CBOW(20000, 50)
  emb_weights = cbow.embeddings.weight.data # shape(20.000, 50)
  lang = Language(emb_weights)
  sentence = torch.tensor([[5, 89, 3]]) # shape(1, 3)
  out = lang(sentence) # shape(1, 7)