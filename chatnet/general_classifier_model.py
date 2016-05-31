# -*- coding: utf-8 -*-
"""
    general_classifier_model.py
    ~~~~~~~~

    General pipeline for classifiers and regressors

    USAGE
    ~~~~~

    Single Classifier:
        pipe = ClassifierPipeline(**super_kwargs)
        pipe.setup(df)
        pipe.run(classifier, classifier_arguments)

    Regression:
        pipe = ClassifierPipeline(**super_kwargs)
        pipe.setup(df)
        pipe.run(regressor, regressor_arguments)

    Multiple Classifiers w/ Voting:
        pipe = ClassifierPipeline(**super_kwargs)
        pipe.setup(df)
        pipe.run([(cl_0, {cl_0_args}), (cl_1, ), ...], VotingClassifier_args)

"""
from chatnet.pipes import Pipeline

from . import logger
from collections import Counter
import math
from scipy import sparse
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import VotingClassifier


def sequence_to_csr(row):
    bow_counts = Counter(row)
    for word_ix, ct in bow_counts.iteritems():
        yield (word_ix, ct)


def create_csr_matrix(x_train, n_symbols, skip_top=3):
    """
    Given training data in form of sequences of word indices
    create sparse matrix of vocab-wide count vectors
    """
    rows = []
    cols = []
    vals = []

    for data_ix, row in enumerate(x_train):
        for word_ix, ct in sequence_to_csr(row):
            if word_ix < skip_top:
                continue
            rows.append(data_ix)
            cols.append(word_ix)
            vals.append(ct)

    return sparse.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(len(x_train), n_symbols)
    )


def create_vsm_matrix(x_train, embeddings):
    rows = []
    for data_ix, row in enumerate(x_train):
        row_data = []
        for word_ix in row:
            row_data.append(embeddings[word_ix])
        rows.append(np.hstack(row_data))

    return np.array(rows)


def train_pca_classifier(learning_data, pca_dims, classifier=SVC, **cl_kwargs):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data

    pca = TruncatedSVD(n_components=pca_dims)
    n_symbols = max(
        np.max(X_train) + 1, np.max(X_test) + 1
    )
    logger.info("Forming CSR Matrices")
    x_train, x_test = create_csr_matrix(X_train, n_symbols), create_csr_matrix(X_test, n_symbols)
    logger.info("Starting PCA")
    # pseudo-supervised PCA: fit on positive class only
    pca = pca.fit(x_train[y_train > 0])

    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    logger.info("Starting Classifier")
    if isinstance(classifier, list):
        cl_list = []
        for i in range(len(classifier)):
            cl_str = "cl" + str(i)
            cl_i_args = {} if not classifier[i][1:] else classifier[i][1:][0]
            cl_list.append((cl_str, classifier[i][0](**cl_i_args)))
        cl = VotingClassifier(cl_list, **cl_kwargs)
    else:
        cl = classifier(**cl_kwargs)
    cl.fit(x_train_pca, y_train)
    logger.info("Scoring Classifier")
    score = cl.score(x_test_pca, y_test)
    logger.info(score)
    cl.test_score = score
    pca.n_symbols = n_symbols
    return cl, pca, x_train_pca, x_test_pca


def get_pca_mats(learning_data, pca):
    (X_train, y_train, train_ids), (X_test, y_test, test_ids) = learning_data
    n_symbols = np.max(X_train) + 1
    x_train, x_test = create_csr_matrix(X_train, n_symbols), create_csr_matrix(X_test, n_symbols)
    if not pca:
        return x_train, x_test
    return pca.transform(x_train), pca.transform(x_test)


class ClassifierPipeline(Pipeline):
    captured_kwargs = {'pca_dims', 'df'}
    persisted_attrs = {'cl', 'pca', 'word_index'}
    def __init__(self, *args, **kwargs):
        super_kwargs = {k: v for k, v in kwargs.iteritems() if k not in self.captured_kwargs}
        super(ClassifierPipeline, self).__init__(**super_kwargs)
        self.pca_dims = kwargs.get('pca_dims', 500)
        if 'df' in kwargs:
            self.setup(kwargs['df'])

    def run(self, classifier=SVC, **cl_kwargs):
        self.cl, self.pca, self.x_train_pca, self.x_test_pca = train_pca_classifier(self.learning_data, self.pca_dims, classifier, **cl_kwargs)
        return self.cl.test_score

    def predict(self, new_df):
        self._set_token_data(new_df)
        self._set_learning_data(test_split=0., max_dummy_ratio=1)
        (X, y, ids), _ = self.learning_data
        if len(X) == 0:
            return None
        x_pca = self.pca.transform(create_csr_matrix(X, self.pca.n_symbols))
        return (self.cl.predict_proba(x_pca), ids)

    def test_and_score(self, new_df):
        self._set_token_data(new_df)
        self._set_learning_data(test_split=0., max_dummy_ratio=1)
        (X, y, ids), _ = self.learning_data
        if len(X) == 0:
            return None
        x_pca = self.pca.transform(create_csr_matrix(X, self.pca.n_symbols))
        return self.cl.score(x_pca, y)
