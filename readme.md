# Chatnet

This is a text manipulation and classification package meant for modeling categorical information about real world text data.

The "chat" aspect comes from its initial application in classifying consumer support chats, though of course it is of broad use.

The "net" part comes from the implementation of a recurrent neural network (RNN) for the modeling, using Keras. A PCA + SVM model from Scikit-learn is also provided with a common interface for a baseline comparison. In the initial application, the RNN had higher test-set performance and lower memory requirements during training, but of course your results may vary!

The common interface above these models provides model serialization/deserialization, as well as a shared text cleaning / vocabulary control system. 


### Example

The package is meant to fit right in with lightly-prepared text data, but with an (overly) opinionated input data shape.

The input data are expected to be in a Pandas DataFrame where:

* the text content is expected to be a list of strings (originally, messages)
* the IDs of each message are also expected to be strings
* the class of training data is also a column


```{python}
from chatnet import svm_model, keras_model

pipe = keras_model.KerasPipeline(df=df, message_key='content', label_col='class', skip_top=0, positive_class='1', embedding_size=50, chunk_size=10)

pipe = svm_model.SVMPipeline(df=df, message_key='content', label_col='class', skip_top=0, positive_class='1', pca_dims=2) 

pipe.predict(new_df)  # returns (predictions, ids)
```

### Helpful features for RNN applications

This package provides several handy utilities for RNN modeling:

* smooth integration with pre-trained word vectors of varying dimension (developed with an eye to the gloVe Twitter dataset)
* chunking of inputs into fixed-word-length sections, managing both padding, maintaining input identifiers, and keeping train/test splits uncorrupted
* modular place for domain-specific logic around text processing for out-of-vocabulary tokens (eg, tokens containing '$' become the special token '#price' rather than the less-informative 'out of vocabulary' token)

Using these utilities, a training data set consisting of 'one class per document' could be used to create a model with 'one class prediction per 10-word chunk', enriching the labeled dataset.
