import numpy as np
from . import logger
import re, string
from collections import Counter

numeric_pat = re.compile('.*[\d].*')
caps_pat = re.compile('.*[A-Z].*')


class TextPrepper(object):
    def __init__(self, exclude=set(string.punctuation), pad_char=0, start_char=1, oov_char=2, index_from=3):
        self.pad_char = pad_char
        self.oov_char = oov_char
        self.start_char = start_char
        self.special_chars = {self.oov_char, self.pad_char, self.start_char}
        self.index_from = len(self.special_chars)
        self.exclude = exclude

    def cleaner(self, word):
        if not numeric_pat.match(word):
            return ''.join(ch for ch in word.lower() if ch not in self.exclude)
        if '$' in word:
            return '$price'
        if '800' in word:
            return '$phone'
        if 'www' in  word or 'http' in word:
            return '$web'
        if '-' in word or caps_pat.match(word):
            return '$model'
        else:
            return '$digit'

    def to_matrices(self, df, word_index, id_col='Chat Session ID', label_col='Chat Type',
                     data_col='msgs', positive_class='product', seed=133, test_split=.2, **kwargs):

        df = df[~df[id_col].isnull()]
        ids = df[id_col]
        if positive_class is "scores": # regression
            labels = df[label_col]
        elif positive_class is "satisfaction": # binary scores (satisfied/unsatisfied) | current cutoff is [3-5] = satisfied
            labels = df[label_col].map(lambda s: 1 if s >= 3 else 0)
        else: # positive_class for binary moderator labels
            labels = df[label_col].map(lambda v: 1 if v == positive_class else 0)
        labels = zip(ids, labels)
        X = df[data_col].tolist()

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)
        test_ix = int(len(X) * (1 - test_split))
        X_train = np.array(X[:test_ix])
        labels_train = labels[:test_ix]

        X_test = np.array(X[test_ix:])
        labels_test = labels[test_ix:]

        X_train, labels_train = self.chunk_convos(X_train, labels_train, word_index, **kwargs)
        X_test, labels_test = self.chunk_convos(X_test, labels_test, word_index, **kwargs)

        y_train = np.array([x[1] for x in labels_train])
        train_ids = [x[0] for x in labels_train]

        y_test = np.array([x[1] for x in labels_test])
        test_ids = [x[0] for x in labels_test]

        return (X_train, y_train, train_ids), (X_test, y_test, test_ids)

    def chunk_convos(self, X, labels, word_index, chunk_size=100,
                     max_dummy_ratio=2, chunk_overlap_ratio=2):
        chunk_X, chunk_labels = [], []
        skipped = 0
        for x, label in zip(X, labels):
            # x is the whole chat
            l = len(x)
            clean_x = map(self.cleaner, x)
            index_representation = [
                word_index[w] + self.index_from
                    if w in word_index
                    else self.oov_char
                    for w in clean_x
                ]

            chunk_idx = 0
            for start in (range(0, l - chunk_size, chunk_size / chunk_overlap_ratio) or [0]):
                chunk_idx += 1
                chunk = [self.start_char] + index_representation[start:start + chunk_size]
                if not chunk:
                    continue
                pad_size = (chunk_size - len(chunk) + 1) #  add a one due to start_char
                padded = [self.pad_char] * pad_size + chunk
                if (
                    sum(c == self.oov_char for c in padded) > ((chunk_size - pad_size) / max_dummy_ratio)
                    or
                    all([c in self.special_chars for c in chunk])
                    ):
                    skipped += 1
                    continue
                else:
                    chunk_X.append(padded)
                    chunk_labels.append(('_'.join([label[0], str(chunk_idx)]), label[1]))
                # break
        logger.info("Skipped %s for excess dummies" % skipped)
        return chunk_X, chunk_labels


def get_word_counts(data, tp):
    word_counts = Counter()

    def increment(word):
        word_counts[tp.cleaner(word)] += 1

    data.map(lambda r: map(increment, r))

    return word_counts


def get_word_index(word_counts, nb_words=15000, skip_top=None, nonembeddable=None):
    skip_top = skip_top or 10

    vocab = []
    for (ix, (w, _)) in enumerate(word_counts.most_common(nb_words + skip_top)):
        if w.startswith('$'):
            if ix < skip_top:
                skip_top += 1
            vocab.append(w)
        elif (not nonembeddable or not w in nonembeddable) and ix > skip_top:
            vocab.append(w)

    return {v: ix for ix, v in enumerate(vocab)}
