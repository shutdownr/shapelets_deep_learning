import numpy as np
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Optional, Union

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
        tokens = [[self.id2token[t] for t in txt] for txt in ids]

        # detokenize texts:
        return [self._decoder.detokenize(txt, self._convert_parentheses) for txt in tokens]

class EncoderBPE:
    def __init__(self, tokenizer:Optional[Tokenizer]=None):
        self.tokenizer = tokenizer
        self.offset    = 0 if tokenizer is None else len(tokenizer.id2token)
        self.pairings  = []

    def fit(self, texts:List[str], counts:Optional[List[int]]=None):
        # fit tokenizer if necessary:
        if self.tokenizer is None:
            self.tokenizer = Tokenizer()
            counts = self.tokenizer.fit(texts)
            self.offset = len(self.tokenizer.id2token)

        # counts must be either passed or set by tokenizer.fit(...)
        assert counts is not None
        
        # add single elements to pairings:
        self.pairings = []

        # tokenize texts:
        ids = self.tokenizer.tokenize(texts)

        while any([len(txt) > 1 for txt in ids]):
            # find most frequent shapelet:
            s = np.argmax(counts)

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
            assert pairing is not None

            # append new shapelet:
            self.pairings.append(pairing)
            shapelet_id = len(counts)

            # update counts:
            counts.append(best_n)
            counts[pairing[0]] -= best_n
            counts[pairing[1]] -= best_n

            # update text:
            for i,j in occurances:
                ids[i][j] = shapelet_id
                ids[i][j+1] = -1

            for i in range(len(ids)):
                ids[i] = [t for t in ids[i] if t >= 0]

    @property
    def shapelets(self) -> List[str]:
        shapelets = []

        # unzip shaplets:
        for t1, t2 in self.pairings:
            s = []

            if t1 >= self.offset: s.extend(shapelets[t1 - self.offset])
            else:                 s.append(t1)

            if t2 >= self.offset: s.extend(shapelets[t2 - self.offset])
            else:                 s.append(t2)

            shapelets.append(s)

        return self.tokenizer.detokenize(shapelets)
    
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

if __name__ == '__main__':
    texts = [
        'The “</w>” token at the end of each word is added to identify a word boundary so that the algorithm knows where each word ends. This helps the algorithm to look through each character and find the highest frequency character pairing. I will explain this part in detail later when we will include “</w>” in our byte-pairs.',
        'Moving on next, we will split each word into characters and count their occurrence. The initial tokens will be all the characters and the “</w>” token.'
    ] 

    e  = EncoderBPE()
    e.fit(texts=texts)
    print(e.shapelets)
    print(e.encode(texts))
    print(e.decode(e.encode(texts)))