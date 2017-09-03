from chatnet.pipes import Pipeline
from chatnet import prep

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adadelta
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler

import os


def get_conv_rnn(embedding_weights, **options):
    """
    Required parameters :
        max_features (default 15001)
        maxlen (default 101)
        embedding_size (default 200)
        filter_length (default 4)
        nb_filter (default 32)
        pool_length (default 3)
        gru_output_size (default 100)
    """

    # Embedding
    max_features = options.get('max_features', 15001)
    maxlen = options.get('maxlen', 101)
    embedding_size = options.get('embedding_size', 200)
    embedding_dropout = options.get('embedding_dropout', .15)

    # Convolution
    filter_length = options.get('filter_length', 4)
    nb_filter = options.get('nb_filter', 64)
    pool_length = options.get('pool_length', 3)

    # gru
    gru_output_size = options.get('gru_output_size', 150)
    gru_dropout = options.get('gru_dropout', .05)
    gru_l2_coef_w = options.get('gru_l2_coef_w', .0001)
    gru_l2_coef_u = options.get('gru_l2_coef_u', .0001)

    # learning
    clipnorm = options.get('clipnorm', 15.)
    n_classes = options.get('n_classes', 1)
    if n_classes == 1:
        loss = 'binary_crossentropy'
        final_activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        final_activation = 'softmax'

    print('Build model...')

    model = Sequential()
    if embedding_weights is not None:
        model.add(Embedding(max_features, embedding_size, input_length=maxlen, weights=[embedding_weights]))
    else:
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(embedding_dropout))
    model.add(Conv1D(filters=nb_filter,
                     kernel_size=filter_length,
                     padding='valid',
                     activation='tanh',
                     strides=1))

    model.add(MaxPooling1D(pool_size=pool_length))
    model.add(Dropout(0.5))
    model.add(GRU(gru_output_size, dropout=gru_dropout, recurrent_dropout=gru_dropout,
                  kernel_regularizer=l2(gru_l2_coef_w), recurrent_regularizer=l2(gru_l2_coef_u)))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dense(n_classes))
    model.add(Activation(final_activation))

    model.compile(loss=loss,
                  optimizer=Adadelta(clipnorm=clipnorm),
                  metrics=['accuracy'])
    return model


def train(model, X_train, y_train, X_test, y_test, **options):
    """
    Kwarg options:
        batch_size (default 32)
        epochs (default 10)
    """
    # Training
    batch_size = options.get('batch_size', 32)
    epochs = options.get('epochs', 10)

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, y_test), callbacks=[LearningRateScheduler(get_rate)])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


class KerasPipeline(Pipeline):
    """Pipeline adapted for convolutional RNN use

    Some parameters must be coordinated between RNN and text prep
    These are set as defaults to keras_model_options
        maxlen, the RNN input sequence length, is chunk_size + 1
        max_features, the size of the embedded vocab, is n_symbols
    For other keras_model_options, see docstring of get_conv_rnn

    """
    captured_kwargs = {'keras_model_options', 'embedding_size', 'df'}
    persisted_attrs = {'word_index'}

    def __init__(self, *args, **kwargs):

        super_kwargs = {k: v for k, v in kwargs.items() if k not in self.captured_kwargs}
        super(KerasPipeline, self).__init__(*args, **super_kwargs)
        self.keras_model_options = kwargs.get('keras_model_options', {})
        self.embedding_size = kwargs.get('embedding_size')

        if 'df' in kwargs:
            self.setup(kwargs['df'])

    def _set_vocabulary(self):
        self.nonembeddable = prep.get_nonembeddable_set(self.word_counts)

        self.set_word_index(skip_top=self.skip_top, nonembeddable=self.nonembeddable)

        self.embedding_weights, self.n_symbols = prep.get_embedding_weights(
            self.word_index, embedding_size=self.embedding_size
        )

        self.keras_model_options.setdefault('maxlen', self.to_matrices_kwargs.get('chunk_size', 100) + 1)
        self.keras_model_options.setdefault('max_features', self.n_symbols)
        self.keras_model_options.setdefault('embedding_size', self.embedding_size)
        self.keras_model_options.setdefault(
            'n_classes', 1 if self.label_mode == 'binary' else self.data[self.label_col].nunique()
        )

        self.model = get_conv_rnn(self.embedding_weights, **self.keras_model_options)

    def run(self, **training_options):
        (X_train, y_train, train_ids), (X_test, y_test, test_ids) = self.learning_data
        train(self.model, X_train, y_train, X_test, y_test, **training_options)

    def predict(self, new_df):
        self._set_token_data(new_df)
        self._set_learning_data(test_split=0., max_dummy_ratio=1, **self.to_matrices_kwargs)
        (X, y, ids), _ = self.learning_data
        predictions = self.model.predict(X)
        return (predictions, ids)

    def persist(self, name, path):
        path_for = lambda attr: os.path.join(path, '_'.join([attr, name]))  # noqa
        super(KerasPipeline, self).persist(name, path)
        model_json = self.model.to_json()
        open(path_for('model_json'), 'w').write(model_json)
        self.model.save_weights(path_for('model_weights'))

    @classmethod
    def restore(cls, name, path):
        path_for = lambda attr: os.path.join(path, '_'.join([attr, name]))  # noqa
        pipe = super(KerasPipeline, cls).restore(cls, name, path)
        model = model_from_json(open(path_for('model_json'), 'r').read())
        model.load_weights(path_for('model_weights'))
        return pipe


def get_rate(epoch):
    if epoch < 3:
        return .01
    if epoch < 6:
        return .005
    return .002
