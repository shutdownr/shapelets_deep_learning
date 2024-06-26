import numpy as np
import json

from sklearn.feature_extraction.text import TfidfTransformer
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

from typing import List, Optional

#====================================================================================================#
# Helpers:                                                                                           #
#====================================================================================================#

def count_ids(ids:List[List[int]], size:int) -> np.ndarray:
    counts = np.zeros((len(ids), size), dtype=int)

    for i, txt in enumerate(ids):
        if len(txt) > 0:
            unique_ids, count_values = np.unique(txt, return_counts=True)
            counts[i, unique_ids] = count_values

    return counts

#====================================================================================================#
# Classes:                                                                                           #
#====================================================================================================#

class Tokenizer:
    def __init__(self, convert_parentheses:bool=False) -> None:
        """Creates a new tokenizer."""
        # Word tokenization:
        self._encoder = TreebankWordTokenizer()
        self._decoder = TreebankWordDetokenizer()
        self._convert_parentheses = convert_parentheses

        # Integer conversion:
        self.id2token = ['<unk>']
        self.token2id = {'<unk>':0}

    def fit(self, texts:List[str]) -> List[int]:
        # tokenize texts:
        tokens = [self._encoder.tokenize(txt) for txt in texts]

        # create vocabulary and word counts:
        self.id2token, counts = np.unique(np.concatenate(tokens), return_counts=True)
        
        # add unknown token:
        self.id2token = ['<unk>'] + list(self.id2token)

        # create dictionary for detokenization:
        self.token2id = {t:i for i,t in enumerate(self.id2token)}

        return [0] + list(counts)

    def tokenize(self, texts:List[str]) -> List[List[int]]:
        # tokenize texts:
        tokens = [self._encoder.tokenize(txt, self._convert_parentheses) for txt in texts]

        # encode texts:
        return [[self.token2id[t] if t in self.token2id else 0 for t in txt] for txt in tokens]

    def detokenize(self, ids:List[List[int]]) -> List[str]:
        # decode texts:
        tokens = [[self.id2token[t] for t in txt if t > 0] for txt in ids]

        # detokenize texts:
        return [self._decoder.detokenize(txt, self._convert_parentheses) for txt in tokens]
    
    def save(self, path:str) -> None:
        # create data dictionary:
        data = {
            'id2token':self.id2token,
            'token2id':self.token2id,
            'convert_parentheses': self._convert_parentheses
        }

        # write dictionary to disk:
        with open(path, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def load(path:str) -> 'Tokenizer':
        # load data dictionary:
        with open(path, 'r') as file:
            data = json.load(file)

        # create tokenizer object:
        tokenizer = Tokenizer(convert_parentheses=data['convert_parentheses'])
        tokenizer.id2token = data['id2token']
        tokenizer.token2id = data['token2id']

        return tokenizer

class EncoderBPE:
    def __init__(self, tokenizer:Optional[Tokenizer]=None):
        self.tokenizer = tokenizer
        self.offset    = 0 if tokenizer is None else len(tokenizer.id2token)
        self.pairings  = []

    def fit(self, texts:List[str], counts:Optional[List[int]]=None, verbose:bool=False):
        # fit tokenizer if necessary:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
            self.tokenizer.fit(texts)
            self.offset = len(self.tokenizer.id2token)

        # tokenize texts:
        ids = self.tokenizer.tokenize(texts)
        counts = count_ids(ids, self.offset)

        # add single elements to pairings:
        self.pairings = []

        while any([len(txt) > 1 for txt in ids]):
            # find most informative shapelet:
            tfidf = TfidfTransformer().fit_transform(counts)
            tfidf = np.asarray(tfidf.sum(axis=0))[0]
            s = np.argmax(tfidf)

            # find the most common shapelet:
            #s = np.argmax(counts.sum(axis=0))

            if verbose:
                print('Counts:', counts.shape, type(counts), counts)
                print('TF-IDF:', tfidf.shape, type(tfidf), tfidf)

            # find all pairs including the shaplet:
            pairings = {}
            for i, txt in enumerate(ids):
                for j, shapelet in enumerate(txt):
                    if shapelet == s:
                        if j > 0:
                            key = (txt[j-1], shapelet)
                            if not key in pairings: pairings[key] = []
                            pairings[key].append((i,j-1))

                        if j + 1 < len(txt):
                            key = (shapelet, txt[j+1])
                            if not key in pairings: pairings[key] = []
                            pairings[key].append((i,j))
 
            # find most frequent pairing:
            pairing, occurances, best_n = None, [], 0
            for key in pairings:
                n = len(pairings[key])
                if n > best_n:
                    pairing, occurances, best_n = key, pairings[key], n
            assert pairing is not None, counts[:,s]

            # append new shapelet:
            self.pairings.append(pairing)
            shapelet_id = counts.shape[1]

            # update text:
            for i,j in occurances:
                ids[i][j] = shapelet_id
                ids[i][j+1] = -1

            for i in range(len(ids)):
                ids[i] = [t for t in ids[i] if t >= 0]
                if len(ids[i]) == 1: ids[i] = []

            # update counts:
            # if pairing[0] == pairing[1]:
            #     counts.append(int(0.5*best_n))
            # else:
            #     counts.append(best_n)
            # counts[pairing[0]] -= best_n
            # counts[pairing[1]] -= best_n

            # ToDo: make more performant:
            counts = count_ids(ids, counts.shape[1] + 1)

    @property
    def shapelets(self) -> List[List[int]]:
        shapelets = []

        # unzip shaplets:
        for t1, t2 in self.pairings:
            s = []

            if t1 >= self.offset: s.extend(shapelets[t1 - self.offset])
            else:                 s.append(t1)

            if t2 >= self.offset: s.extend(shapelets[t2 - self.offset])
            else:                 s.append(t2)

            shapelets.append(s)

        return shapelets

    @property
    def shapelets_decoded(self) -> List[str]:
        return self.tokenizer.detokenize(self.shapelets)

    def get_filtered_shapelets(self, min_length:int=2, max_length:int=10, pad=True, **kwargs) -> List[List[int]]:
        '''Filters the shapelets based on a min and max length'''
        filtered_shapelets = [s for s in self.shapelets if len(s)<=max_length and len(s)>=min_length]

        if pad:
            padded = np.full((len(filtered_shapelets),max_length),-1,dtype=int)
            for i, s in enumerate(filtered_shapelets):
                padded[i,:len(s)] = s
            return padded

        return filtered_shapelets

    def encode(self, texts:List[str]) -> List[List[int]]:
        # tokenize text:
        ids = self.tokenizer.tokenize(texts)

        # exchange ids with higher-order ids:
        for p, (t0, t1) in enumerate(self.pairings):
            p += self.offset
            for i in range(len(ids)):

                j = 0
                while j < (len(ids[i]) - 1):
                    j += 1

                    if ids[i][j-1] != t0: continue
                    if ids[i][j]   != t1: continue

                    ids[i] = ids[i][:j-1] + [p,] + ids[i][j+1:]

        return ids

    def decode(self, ids:List[List[int]]) -> List[str]:
        # exchange ids with lower-order ids:
        offset = len(self.pairings) + self.offset - 1
        for p, (t0, t1) in enumerate(reversed(self.pairings)):
            p = offset - p
            for i in range(len(ids)):

                j = 0
                while j < len(ids[i]):
                    j += 1

                    if ids[i][j-1] != p: continue


                    ids[i] = ids[i][:j-1] + [t0, t1] + ids[i][j:]
                    j += 1

        # detokenize ids:
        return self.tokenizer.detokenize(ids)

class EncoderTfIdf:
    def __init__(self, tokenizer:Optional[Tokenizer]=None):
        self.tokenizer   = tokenizer
        self.transformer = TfidfTransformer()
        self.vocab_size  = len(self.tokenizer.id2token)

    def fit(self, texts:List[str], counts:Optional[List[int]]=None):
        # fit tokenizer if necessary:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
            counts = self.tokenizer.fit(texts)
            self.vocab_size = len(self.tokenizer.id2token)

        # tokenize texts:
        ids = self.tokenizer.tokenize(texts)

        # fit tfidf:
        counts = count_ids(ids, self.vocab_size)
        #print(counts.shape, type(counts), counts)
        self.transformer.fit(counts)

    def encode(self, texts:List[str]) -> List[List[int]]:
        # tokenize text:
        ids = self.tokenizer.tokenize(texts)

        # calculate TfIdf:
        counts = count_ids(ids, self.vocab_size)
        #print(counts.shape, type(counts), counts)

        return self.transformer.fit_transform(counts).toarray()

#====================================================================================================#
# Functions:                                                                                         #
#====================================================================================================#

def shapelet_transform_text(X:np.ndarray, shapelets:np.ndarray, tokenizer: Tokenizer):
    # TODO: Implement
    # Should return an ndarray of float features of the shape (N,m)
    # Where N is the number of instances and m is the output feature dimensionality
    features = np.zeros((len(X), len(shapelets)), dtype=float)

    # For each sample
    for i, txt in enumerate(tokenizer.tokenize(X)):
        # For each discovered shapelet candidate
        for j, shapelet in enumerate(shapelets):
            shapelet = shapelet[shapelet>0]
            features[i,j] = np.inf
            s_length = len(shapelet)
            for k in range(len(txt)-s_length+1):
                distance = np.mean(txt[k:k+s_length] != shapelet.astype(float))
                if distance < features[i,j]:
                    features[i,j] = distance

    return features