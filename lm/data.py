import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = {}
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter.setdefault(word, 0)
        self.counter[word] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        import pickle
        name = path.split('/')[1]
        if os.path.exists('dictionary_' + name):
            with open('dictionary_' + name, 'rb') as file:
                self.dictionary = pickle.load(file)
        else:
            self.tokenize(os.path.join(path, 'train.txt'))
            self.tokenize(os.path.join(path, 'valid.txt'))
            self.tokenize(os.path.join(path, 'test.txt'))
            new_dict = [(self.dictionary.counter[i], i) for i in self.dictionary.word2idx]
            new_dict.sort(key=lambda x: x[0])
            new_dict.reverse()
            for i in range(len(new_dict)):
                self.dictionary.word2idx[new_dict[i][1]] = i
                self.dictionary.idx2word[i] = new_dict[i][1] 
                self.dictionary.counter[i] = new_dict[i][0]
            with open('dictionary_' + name, 'wb') as file:
                pickle.dump(self.dictionary, file)
        #print(self.dictionary.word2idx)
        self.train = self.tokenize_(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize_(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize_(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        return
        
    def tokenize_(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
        #        for word in words:
        #            self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


