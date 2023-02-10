# From https://github.com/pytorch/examples/blob/d8456a36d1bbb22f72b003f59406a19a0a0547c3/word_language_model/data.py

import os
from io import open
import torch

VOCAB_SIZE = 267735

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Tokenizer(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.path = path
    
    def tokenize_wikitext103(self):
        self.train_data = self.tokenize(os.path.join(self.path, 'wiki.train.tokens'))
        self.valid_data = self.tokenize(os.path.join(self.path, 'wiki.valid.tokens'))
        self.test_data = self.tokenize(os.path.join(self.path, 'wiki.test.tokens'))

    def train(self):
        """Tokenizes a text file."""
        path = os.path.join(self.path, 'wiki.train.tokens')
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path):
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids

    def save_vocab(self, path):
        if os.path.exists(path):
            raise IOError(f"File already exists {path}.")
        with open(path, 'w', encoding="utf8") as f:
            for word in self.dictionary.idx2word:
                f.write('\n')
                f.write(word)
    
    def load_vocab(self, path):
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                if line == '\n':
                    continue
                word = line.split()[0]
                self.dictionary.add_word(word)