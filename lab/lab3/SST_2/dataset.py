import os
import random
import string
import nltk
import pickle

class basedataset():
    def __init__(self, mode, shuffle=False, maxlen=None):
        assert mode in ['train', 'test', 'dev']
        self.root = './SST_2/'+mode+'.tsv'
        f = open(self.root, 'r', encoding='utf-8')
        L = f.readlines()
        self.data = [x.strip().split('\t') for x in L]
        if maxlen is not None:
            self.data = self.data[:maxlen]
        self.len = len(self.data)
        self.D = []
        for i in range(self.len):
            self.D.append(i)
        if shuffle:
            random.shuffle(self.D)
        self.count = 0

    def tokenize(self, text):
        cleaned_tokens = []
        tokens = nltk.tokenize.word_tokenize(text.lower())
        for token in tokens:
            if token in nltk.corpus.stopwords.words('english'):
                continue
            else:
                all_punct = True
                for char in token:
                    if char not in string.punctuation:
                        all_punct = False
                        break
                if not all_punct:
                    cleaned_tokens.append(token)
        return cleaned_tokens

    def __getitem__(self, index, show=False):
        index = self.D[index]
        text, label = self.data[index]
        tokenize_text = text.strip()
        tokenize_text = self.tokenize(tokenize_text)
        if show == True:
            return (tokenize_text, text), int(label)
        else:
            return tokenize_text, int(label)
    
    def get(self, index):
        index = self.D[index]
        text, label = self.data[index]
        return text, int(label)

def traindataset(shuffle=False):
    return basedataset('train', shuffle)

def minitraindataset(shuffle=False):
    return basedataset('train', shuffle, maxlen=128)

def testdataset(shuffle=False):
    return basedataset('dev', shuffle=False)

def validationdataset(shuffle=False):
    return basedataset('dev', shuffle=False)

if __name__ == '__main__':
    pass