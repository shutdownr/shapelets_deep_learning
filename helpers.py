import numpy as np
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from typing import List, Tuple, Optional, Union

class Tokenizer:
    def __init__(self, convert_parentheses:bool=False) -> None:
        """Creates a new tokenizer."""
        # Word tokenization:
        self._encoder = TreebankWordTokenizer()
        self._decoder = TreebankWordDetokenizer()
        self._convert_parentheses = convert_parentheses

        # Integer conversion:
        self.id2token = []
        self.token2id = {}

    def fit(self, texts:List[str]) -> Union[List[int], np.ndarray]:
        # tokenize texts:
        tokens = [self._encoder.tokenize(txt) for txt in texts]

        # create vocabulary and word counts:
        self.id2token, counts = np.unique(np.concatenate(tokens), return_counts=True)

        # create dictionary for detokenization:
        self.token2id = {t:i for i,t in enumerate(self.id2token)}

        return counts

    def tokenize(self, texts:List[str]) -> List[List[int]]:
        # tokenize texts:
        tokens = [self._encoder.tokenize(txt, self._convert_parentheses) for txt in texts]

        # encode texts:
        return [[self.token2id[t] for t in txt] for txt in tokens]

    def detokenize(self, ids:List[List[int]]) -> List[str]:
        # decode texts:
        tokens = [[self.id2token[t] for t in txt] for txt in ids]

        # detokenize texts:
        return [self._decoder.detokenize(txt, self._convert_parentheses) for txt in tokens]

def encode_bpe(ids:List[List[int]], counts:List[int]):
    counts    = list(counts)
    shapelets = [(i,) for i in range(len(counts))]

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
        shapelets.append(pairing)
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

    # unravel shaplets:
    unravelled = []

    while len(shapelets) > 0:
        s = shapelets.pop(0)

        s_new = []
        for t in s:
            if t < len(unravelled):
                s_new += unravelled[t]
            else:
                s_new.append(t)

        unravelled.append(s_new)

    return unravelled

if __name__ == '__main__':
    texts = [
        'The “</w>” token at the end of each word is added to identify a word boundary so that the algorithm knows where each word ends. This helps the algorithm to look through each character and find the highest frequency character pairing. I will explain this part in detail later when we will include “</w>” in our byte-pairs.',
        'Moving on next, we will split each word into characters and count their occurrence. The initial tokens will be all the characters and the “</w>” token.'
    ] 

    t  = Tokenizer()

    counts = t.fit(texts=texts)
    ids    = t.tokenize(texts)

    print(ids)
    print(t.detokenize(ids))

    print(encode_bpe(ids, counts))