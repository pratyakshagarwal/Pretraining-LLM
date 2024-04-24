import pandas as pd
import torch

text = open('input.txt', 'r', encoding='utf-8').read()

vocab = sorted(list(set(text)))  # Unique characters in the text
stoi = {ch: i for i, ch in enumerate(vocab)}  # Mapping from character to index
itos = {i: ch for ch, i in stoi.items()}  # Mapping from index to character

# Functions for encoding and decoding
encode = lambda string: [stoi[ch] for ch in string]  # Encoding function
decode = lambda seq: ''.join(itos[x] for x in seq)  # Decoding function

# Encode the whole text
data = encode(text)

# Total length of the data
total_len = len(data)

# Split data into train and validation sets
n = int(0.9 * total_len)  # 90% of the data for training
train_data = data[:n]  # First 90% of the data for training
val_data = data[n:]  # Remaining 10% of the data for validation


def get_batch(split):
    """
    Generate a small batch of data for inputs (x) and targets (y) based on the split (train/validation).

    Args:
        split (str): 'train' for training batch, 'val' for validation batch.

    Returns:
        torch.Tensor: Input tensor x of shape (batch_size, block_size).
        torch.Tensor: Target tensor y of shape (batch_size, block_size).
    """
    # Select data based on the split
    data = train_data if split == 'train' else val_data

    # Generate random indices for the batch
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # Create input x and target y batches
    x = torch.stack([torch.tensor(data[i:i+config.block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+config.block_size+1]) for i in ix])

    # Move tensors to the specified device (CPU or GPU)
    x, y = x.to(config.device), y.to(config.device)

    return x, y
