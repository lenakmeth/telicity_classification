#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:30:22 2021

@author: lena
"""

import numpy as np
from tqdm import tqdm_notebook
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import torch
import torch.nn as nn
import torch.nn.functional as F


def open_vocab(label_marker):
    vocab = []
    with open(label_marker + '/vocab', 'r', encoding='utf-8') as f:
        vocab += [line.strip() for line in f]
    vocab = {word:idx for idx, word in enumerate(vocab, 2)}
    vocab[1] = '<unk>'
    vocab[0] = '<pad>'
    
    return vocab

def encode_sequence(sent, vocab):
        
    encoded = []
    for word in sent.split(' '):
        if word in vocab:
            encoded.append(vocab[word])
        else:
            encoded.append(1)
    return encoded

def make_sets(type_set, label_marker, max_seq):
    
    vocab = open_vocab(label_marker)
   
    X = []
    y = []

    with open(label_marker + '/' + type_set, 'r', encoding='utf-8') as f:
        for line in f:
            encoded_seq = encode_sequence(line.split('\t')[-4], vocab)
            
            if len(encoded_seq) < max_seq:
                encoded_seq = encoded_seq + [0]*(max_seq-len(encoded_seq))
            else:
                encoded_seq = encoded_seq[:max_seq]
            
            assert len(encoded_seq) == max_seq
            
            X.append(encoded_seq)
            y.append(int(line.strip().split('\t')[-1]))

    return np.asarray(X), np.asarray(y)


def load_pretrained_vectors(word2idx, idx2word, fname):
    """Load pretrained vectors and create embedding layers.
    
    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())

    # Initilize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0].strip()
        if word in idx2word:
            count += 1
            embeddings[idx2word[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings



def data_loader(train_inputs, val_inputs, train_labels, val_labels,
                batch_size=50):
    """Convert train and validation sets to torch.Tensors and load them to
    DataLoader.
    """

    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels =\
    tuple(torch.LongTensor(data) for data in
          [train_inputs, val_inputs, train_labels, val_labels])

    # Specify batch_size
    batch_size = 50

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader

def predicted_label(encoded_sent, model, max_len=62):
    """Predict probability that a review is positive."""

        # Convert to PyTorch tensors
    input_id = torch.tensor(encoded_sent).unsqueeze(dim=0)

    # Compute logits
    logits = model.forward(input_id)

    #  Compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    
    if probs[0] > 0.5:
        return 0
    else:
        return 1
    