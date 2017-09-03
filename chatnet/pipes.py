from . import logger
from chatnet import prep
import pandas as pd
from collections import Counter
from sklearn.externals import joblib
import os


class Pipeline(object):
    """
    Transformer helper functions and state checkpoints
    to go from text data/labels to model-ready numeric data


    """
    def __init__(self, vocab_size=15000,
                 data_col=None, id_col=None, label_col=None, skip_top=10,
                 positive_class='product', df=None, message_key=None, **kwargs
                 ):

        # message processing
        self.data_col = data_col or 'tokens'
        self.id_col = id_col or 'id'
        self.label_col = label_col or 'labels'
        self.message_key = message_key or 'msgs'

        self.positive_class = positive_class
        if positive_class is None:
            self.label_mode = 'multiclass'
            self.n_classes = []
        else:
            self.label_mode = 'binary'

        # vocab processing
        self.tp = prep.TextPrepper()
        self.vocab_size = vocab_size
        self.skip_top = skip_top
        self.to_matrices_kwargs = kwargs

        if df is not None:
            self.setup(df)

    def _tokenize(self, df, message_key=''):
        """
        Iterate over each row's messages (as specified by message_key),
        tokenizing by ' ' and cleaning with self.tp.cleaner
        """
        if isinstance(message_key, (unicode, str)):
            message_generator = get_message_generator(message_key, kind='dense')
        elif isinstance(message_key, list):
            message_generator = get_message_generator(message_key, kind='wide')

        def mapper(row):
            sequence = []
            for message in message_generator(row):
                sequence += map(self.tp.cleaner, message.split())
            return sequence

        df[self.data_col] = df.apply(mapper, axis=1)

    def _set_token_data(self, input_df):
        df = input_df.copy()
        if self.data_col not in df.columns:
            self._tokenize(df, message_key=self.message_key)

        self.data = pd.DataFrame(df[[self.data_col, self.id_col, self.label_col]])

        logger.info("Counting words...")
        self.set_word_counts()

    def _set_vocabulary(self):
        # This is extended by subclasses with special concerns about word_index (eg word embeddings)
        self.set_word_index(skip_top=self.skip_top)

    def _set_learning_data(self, **to_matrices_kwargs):
        to_matrices_kwargs.setdefault('seed', 212)
        to_matrices_kwargs.setdefault('test_split', .18)
        to_matrices_kwargs.setdefault('chunk_size', 100)
        to_matrices_kwargs.setdefault('data_col', self.data_col)
        to_matrices_kwargs.setdefault('id_col', self.id_col)
        to_matrices_kwargs.setdefault('label_col', self.label_col)
        to_matrices_kwargs.setdefault('positive_class', self.positive_class)
        logger.info("Making numeric sequences...")

        self.learning_data = (X_train, y_train, train_ids), (X_test, y_test, test_ids) = \
            self.tp.to_matrices(self.data, self.word_index, **to_matrices_kwargs)

    def setup(self, df):
        self._set_token_data(df)
        self._set_vocabulary()
        self._set_learning_data(**self.to_matrices_kwargs)

    def set_word_counts(self):
        """
        Map :tp.cleaner over token lists in :data
        and return a counter of cleaned :word_counts
        """
        word_counts = Counter()

        def increment(word):
            word_counts[word] += 1

        self.data[self.data_col].map(lambda r: map(increment, r))

        self.word_counts = word_counts

    def set_word_index(self, skip_top=None, nonembeddable=None):
        """
        Accepts a dictionary of word counts
        Selects the top :nb_words, after skipping the :skip_top most common
        Optionally provide a set of words you don't have word vectors and want to omit entirely
        Always includes special words (returned by self.cleaner) prepended with $

        Returns dict like {word: ranking by count}
        """

        skip_top = 10 if skip_top is None else skip_top

        vocab = []
        for (ix, (w, _)) in enumerate(self.word_counts.most_common(self.vocab_size)):
            if w.startswith('$'):
                if ix < skip_top:
                    skip_top += 1
                vocab.append(w)
            elif (not nonembeddable or w not in nonembeddable) and ix > skip_top:
                vocab.append(w)

        self.word_index = {v: ix for ix, v in enumerate(vocab)}

    def persist(self, name, path):
        for attr in self.persisted_attrs:
            joblib.dump(getattr(self, attr), os.path.join(path, '_'.join([attr, name])))

    @classmethod
    def restore(cls, name, path):
        pipe = cls()
        for attr in cls.persisted_attrs:
            setattr(pipe, attr, joblib.load(os.path.join(path, '_'.join([attr, name]))))
        return pipe


def get_message_generator(message_key, kind='wide'):
    if kind == 'wide':
        # iterate over columns in message_key yielding from row
        def message_generator(row):
            for key in message_key:
                yield row[key]
    elif kind == 'dense':
        # iterate over array of messages in row[message_key]
        def message_generator(row):
            for cell in row[message_key]:
                yield cell
    return message_generator
