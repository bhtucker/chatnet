"""
Functionality for cleaning text inputs from the wild,
constructing fixed-length word index sequences,
and providing embeddings from GloVe word vectors
"""

import numpy as np
from . import logger
import re
import string
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

numeric_pat = re.compile('.*[\d].*')
caps_pat = re.compile('.*[A-Z].*')
GLOVE_VEC_TEMPLATE = '/Users/bhtucker/rc/chatnet/glove.twitter.27B/glove.twitter.27B.{dimensions}d.txt'
GLOVE_DIMS = {25, 50, 200}


class TextPrepper(object):
    def __init__(self, exclude=set(string.punctuation), pad_char=0, start_char=1, oov_char=2, index_from=3):
        self.pad_char = pad_char
        self.oov_char = oov_char
        self.start_char = start_char
        self.special_chars = {self.oov_char, self.pad_char, self.start_char}
        self.index_from = len(self.special_chars)
        self.exclude = exclude

    def cleaner(self, word):
        """
        Return :word with:self.exclude chars (ie punctuation) stripped from token
        Group numeric inputs by ad hoc logic
        Values returned prepended with '$' are always included in vocabulary (see get_word_index)
        """
        if not numeric_pat.match(word):
            return ''.join(ch for ch in word.lower() if ch not in self.exclude)
        if '$' in word:
            return '$price'
        if '800' in word:
            return '$phone'
        if 'www' in word or 'http' in word:
            return '$web'
        if '-' in word or caps_pat.match(word):
            return '$model'
        else:
            return '$digit'

    def to_matrices(self, df, word_index, id_col='Chat Session ID', label_col='Chat Type',
                    data_col='msgs', positive_class='product', seed=133, test_split=.2, **kwargs):
        """
        Using :df and :word_index, return training and test data

        Chunking and filtering will alter the number of rows.
        :kwargs are passed along directly to self.chunk_tokens
        """
        df = df[~df[id_col].isnull()]
        ids = df[id_col]

        if positive_class is None:
            le = LabelEncoder()
            labels = np_utils.to_categorical(le.fit_transform(df[label_col]))
        else:
            labels = df[label_col].map(lambda v: 1 if v == positive_class else 0)
        labels = zip(ids, labels)
        X = df[data_col].tolist()

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)

        # Split data before chunking to keep full observations separate
        test_ix = int(len(X) * (1 - test_split))
        X_train = np.array(X[:test_ix])
        labels_train = labels[:test_ix]

        X_test = np.array(X[test_ix:])
        labels_test = labels[test_ix:]

        X_train, labels_train = self.chunk_tokens(X_train, labels_train, word_index, **kwargs)
        X_test, labels_test = self.chunk_tokens(X_test, labels_test, word_index, **kwargs)

        y_train = np.array([x[1] for x in labels_train])
        train_ids = [x[0] for x in labels_train]

        y_test = np.array([x[1] for x in labels_test])
        test_ids = [x[0] for x in labels_test]

        return (X_train, y_train, train_ids), (X_test, y_test, test_ids)

    def chunk_tokens(self, X, labels, word_index, chunk_size=100,
                     max_dummy_ratio=2, chunk_overlap_ratio=2):
        """
        Neural network models require dense, fixed width inputs
        This function splits up and pads input text into :chunk_size length
        lists of vocabulary index ints. Each observation's labels are duplicated
        in parallel so that the returned numeric chunks representations are labeled

        :X is a list of token lists
        :labels is a list like (id, class label)
        :word_index is a cleaned word : integer index vocabulary dict
        :chunk_size is the guaranteed length of each row in return value :chunk_X
        :max_dummy_ratio is a filter to remove messages with too many out of vocab chars
            if a chunk has over (1/max_dummy_ratio) oov chars, it will not be returned
        :chunk_overlap_ratio is the amount of overlapping text among sequential chunks
            overlapping text gives the model more perspectives on sequences, and more training data
            along with dummy filtering, you can select for higher signal inputs

        Returns:
            :chunk_X, :chunk_labels as the model-ready observations
        """
        chunk_X, chunk_labels = [], []
        skipped = 0
        for x, label in zip(X, labels):
            # x is the whole observation
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
                pad_size = (chunk_size - len(chunk) + 1)  # add a one due to start_char
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

        logger.info("Skipped %s for excess dummies" % skipped)
        return chunk_X, chunk_labels


def get_embedding_weights(word_index, index_from=3, embedding_size=200, zero_pad=0):
    """
    Read through GloVe word vectors for :embedding_size dimensions to get word embeddings
    index_from specifies how many rows should be left for special characters
    Returns:
    :embedding_weights stack of word embedding rows keyed by :word_index + :index_from
    :n_symbols number of rows of embedding_weights
    """
    n_symbols = len(word_index) + index_from
    embedding_weights = np.zeros((n_symbols, embedding_size + zero_pad))

    def update_weights(line):
        if line[:line.find(' ')] in word_index:
            tokens = line.split()
            embedding_weights[word_index[tokens[0]] + index_from] = \
                np.array(map(np.float, tokens[1:] + [0] * zero_pad))

    with open(get_vec_file(embedding_size), 'r') as f:
        map(update_weights, f)

    return embedding_weights, n_symbols


def get_vec_file(dimensions):
    return GLOVE_VEC_TEMPLATE.format(dimensions=dimensions)


def get_nonembeddable_set(word_counts, rank_cutoff=50000):
    """
    Returns set of top :rank_cutoff words without GloVe embeddings
    """
    full_vocab = set(w for w, _ in word_counts.most_common(rank_cutoff))
    seen = set()
    with open(get_vec_file(list(GLOVE_DIMS)[0]), 'r') as f:
        for line in f:
            if line[:line.find(' ')] in full_vocab:
                seen.add(line[:line.find(' ')])
    return full_vocab - seen
