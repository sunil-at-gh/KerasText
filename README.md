# KerasText: Text Processing Layers for Keras

This package adds [Keras](https://keras.io) ver. 2+ compliant layers for processing variable-sized text input, with support for masking.

## Compatibility

- **Python**: ver. 3+.
- [**Keras**](https://keras.io): ver. 2+.
- *Backends*:
    - [**Theano**](http://deeplearning.net/software/theano/): tested with ver. 0.9.0
    - [**TensorFlow**](https://www.tensorflow.org): tested with ver. 1.3.0
    - **CNTK**: not supported.


## Text Document Representation

### K-Dimensional Tensors

The smallest unit in a document is a word (aka token). Words are represented as positive integer indices into a 
Vocabulary, with the special index `0` being used to represent padding. 

For this package, a document is represented as a tensor of arbitrary dimensionality. The simplest document is a 
single sentence or sequence of words, represented as a vector of integers. To represent a document as a sequence
of sentences, a 2-dimensional tensor is used. And so on. We assume that the last dimension of a document is a
sequence of words (a "sentence"), each word denoted by an integer.

A batch of documents adds one more dimension, with the first dimension or axis used to represent the number of 
documents in a batch. The layers in this package will support any number of dimensions.

The tensor shape for a batch of documents where each document is a single sentence is `(nbr_docs, nbr_words)`.
Similarly, when a document is a sequence of sentences, the batch shape is `(nbr_docs, nbr_sentences, nbr_words)`,
where `nbr_words` is the maximum number of words in a sentence, and `nbr_sentences` is the maximum number of 
sentences in a document.

## Right-padding and Masking

Processing text requires handling variable-length documents, and at the lowest level variable-length sentences. For the
layers in this package, all text sequences are assumed to be right-padded with 0's so that a batch of documents can be 
represented as a tensor of some fixed dimensions.

A mask is used to indicate which parts of a document represent valid words, and which parts represent padding. Since
any padding in sentences only occurs in the suffix, the mask for a sentence of width `N` can be represented as the 
vector `1^{m} 0^{n}`, where `m >= 0, n >= 0, m + n = N`. A `1` (or `True`) in the mask denotes a valid word at the
corresponding index, and a `0` (or `False`) indicates padding.

All the layers in this package support text masks of the type described above.


## Text Layers

### Embeddings

The Keras Embedding layer supports inputs of arbitrary dimension, and will translate a K-dimensional document of
integer word indices into a (K+1) dimensional document of word embeddings. When working with batches of documents,
a (K + 1) dimensional batch input results in an output of (K + 2) dimensions. Setting the Embedding layer's argument
`mask_zero = True` will also cause it to transmit an appropriate mask to the layers connected to its output.


### Convolutions and Pooling

The `KTextConvolution` layer supports performing 1-dimensional masked convolution on K-dimensional documents of 
embeddings. This package also provides layers for windowed pooling (e.g. `KTextMaxPooling1D`) and global pooling
(e.g. `KTextGlobalMaxPooling`).


## A Simple Example

Consider a batch of two documents:

```python
docs = [['the first sentence in doc one', 'doc one sentence two'],
        ['the only sentence']]
```

This batch can be represented as a tensor of shape `(nbr_docs = 2, nbr_sentences = 2, nbr_words = 7)`, with
the appropriate right-padding as shown (along with the corresponding mask):

```python
inputs = array([[[7, 2, 6, 3, 1, 4, 0],
                [1, 4, 6, 8, 0, 0, 0]],

               [[7, 5, 6, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]])

mask = array([[[1, 1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 0, 0, 0]],
       
              [[1, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]]])
```

The following script shows a simple model that takes this as input, converts words to embeddings, and then performs a 
convolution.

```python
from keras.layers import Input, Embedding
from keras.models import Model
from kerastext import KTextConvolution
 
def build_model(nbr_sentences, nbr_words, word_vectors, nbr_filters, filter_width):
    vocabulary_sz,  word_vec_sz = word_vectors.shape
    
    input_tensor = Input(shape=(nbr_sentences, nbr_words), dtype='int32', name='input_tensor')
    emb_lyr = Embedding(vocabulary_sz, word_vec_sz, weights=[word_vectors], trainable=False, mask_zero=True)
    
    border_mode = 'valid'
    stride = 1
    activation_func = 'tanh'
    conv_layer = KTextConvolution(filters=nbr_filters, kernel_size=filter_width, padding=border_mode,
                                  strides=stride, activation=activation_func)
                                  
    output_tensor = emb_lyr(input_tensor)
    output_tensor = conv_layer(output_tensor)
    
    return Model(inputs=[input_tensor], outputs=output_tensor)
```
