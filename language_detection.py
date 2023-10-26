import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import pandas as pd 
import pyarrow as pa

# Load Parquet data
data = pd.read_parquet('C:\MLX3\datasets\language_detector\CL_es-en.parquet')["es"]

# Hyperparameters
# VOCAB_SIZE = 120000  # This should be set based on your actual dataset
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_EPOCHS = 2
OUTPUT_DIM = 7  # 7 languages

vocab_size = 50

#vocab_size = data.shape[0]
# Tokenization
tokenizer = get_tokenizer('basic_english')  # You can use a more advanced tokenizer if needed

tokenized_data = []
max_words = 0
for sentence in data:
    tokenized_row = tokenizer(sentence.lower())
    tokenized_data.append(tokenized_row)
    num_of_words = len(tokenized_row) 
    if ( num_of_words > max_words ):
        max_words = num_of_words
    

# Create a random vocabulary for demonstration (In practice, build it from your dataset)
vocab = {word: i for i, word in enumerate(["<pad>", "<unk>"] + list(set(word for sentence in tokenized_data for word in sentence)))}

# Convert tokens to indices based on the vocabulary
indexed_data = [[vocab.get(token, vocab["<unk>"]) for token in sentence] for sentence in tokenized_data]

def pad_array(arr, fixed_size, pad_value=0):
    """
    Pad the input array with the pad_value to make it of the fixed_size.
    
    Parameters:
    - arr (list): The input array to be padded.
    - fixed_size (int): The desired size of the output array.
    - pad_value (str, optional): The value to pad the array with. Default is "<pad>".

    Returns:
    - list: The padded array.
    """   
    # Calculate the number of pad_values needed
    num_pads = fixed_size - len(arr)
    # Return the array padded with the pad_value
    return arr + [pad_value] * num_pads

padded_indexed_data = [pad_array(row, max_words) for row in indexed_data]
print(padded_indexed_data)

# Model definition
class LanguageClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LanguageClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab["<pad>"])
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        # Aggregating embeddings: Average across the sequence dimension to get a single vector for each sentence
        # sentence_embedding = embedded.mean(dim=1)  # [batch_size, embedding_dim]
        
        hidden_output = torch.relu(self.hidden(embedded))  # [batch_size, hidden_dim]
        final_output = self.output(hidden_output)  # [batch_size, output_dim]

          # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(final_output, dim=1)
        
        return probabilities
    

# Create a model instance
model = LanguageClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Sample forward pass (transform a batch of sentences into output predictions)
sample_input = torch.tensor(padded_indexed_data)  # Taking the first two sentences as a sample batch
output = model(sample_input)
#print(output:2)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

labels = []
for i in range(max_words):
    labels.append(0)

# Assume `train_data` is a list of indexed sentences and `train_labels` is a list of language labels
for epoch in range(NUM_EPOCHS):
    for sentence in padded_indexed_data:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        probabilities = model(torch.tensor(sentence))

        # Compute the loss
        loss = loss_function(torch.tensor(probabilities,requires_grad=True), torch.tensor(labels))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

#Test the model
