# codeing: utf-8

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
import gluonnlp as nlp

# class MaskLayer(HybridBlock):
#     def __init__(self):
#         super(MaskLayer, self).__init__()
#
#     def forward(self, mask):
#
#         return (x - nd.min(x)) / (nd.max(x) - nd.min(x))


class CNNTextClassifier(HybridBlock):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    filter_widths : list of int, default = [3,4]
        The widths for each set of filters
    num_filters : int, default = 100
        Number of filters for each width
    num_conv_layers : int, default = 3
        Number of convolutional layers (conv + pool)
    intermediate_pool_size: int, default = 3
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=2, dr=0.2,filter_widths=[3,4],
                 num_filters=100, num_conv_layers=3, intermediate_pool_size=3):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim=emb_input_dim, output_dim=emb_output_dim)
        self.concurrent = gluon.contrib.nn.HybridConcurrent(axis=2)
        self.concurrent2 = gluon.contrib.nn.HybridConcurrent(axis=2)

        with self.name_scope():
            # define layers
            self.localPooling= nn.AvgPool2D(pool_size=(2,1))
            self.GlobalPooling = nn.GlobalMaxPool2D()
            self.drop = nn.Dropout(dr)
            self.out = nn.Dense(num_classes)

            for width in filter_widths:
                self.mlp = nn.HybridSequential()
                self.mlp.add(nn.Conv2D(channels=num_filters, kernel_size=(width, emb_output_dim), activation="relu"))
                self.mlp.add(self.localPooling)
                self.concurrent.add(self.mlp)

            for width in filter_widths:
                self.mlp = nn.HybridSequential()
                self.mlp.add(nn.Conv2D(channels=num_filters, kernel_size=(width, 1), activation="relu"))
                self.concurrent2.add(self.mlp)

            self.concurrent2.add(self.GlobalPooling)
            self.concurrent2.add(self.drop)

    def from_embedding(self, F, embedded, mask):
        reshape = mx.nd.reshape(mask, shape=(mask.shape[0], mask.shape[1], 1) )
        result = mx.nd.broadcast_mul(embedded, reshape)

        result = F.expand_dims(result, axis=1)
        result = self.concurrent(result)
        result = self.concurrent2(result)
        result = self.out(result)
        return result
        
    def hybrid_forward(self, F, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(F, embedded, mask)


class BertCNNTextClassifier(CNNTextClassifier):

    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=2, dr=0.2,filter_widths=[3,4],
                 num_filters=100, num_conv_layers=3, intermediate_pool_size=3):
        super(CNNTextClassifier, self).__init__()
        self.concurrent = gluon.contrib.nn.HybridConcurrent(axis=2)
        self.concurrent2 = gluon.contrib.nn.HybridConcurrent(axis=2)

        with self.name_scope():
            # define layers
            self.localPooling = nn.AvgPool2D(pool_size=(2,1))
            self.GlobalPooling = nn.GlobalMaxPool2D()
            self.drop = nn.Dropout(dr)
            self.out = nn.Dense(num_classes)

            for width in filter_widths:
                self.mlp = nn.HybridSequential()
                self.mlp.add(nn.Conv2D(channels=num_filters, kernel_size=(width, emb_output_dim), activation="relu"))
                self.mlp.add(self.localPooling)
                self.concurrent.add(self.mlp)

            for width in filter_widths:
                self.mlp = nn.HybridSequential()
                self.mlp.add(nn.Conv2D(channels=num_filters, kernel_size=(width, 1), activation="relu"))
                self.mlp.add(self.drop)
                self.concurrent2.add(self.mlp)

            self.concurrent2.add(self.GlobalPooling)
            self.concurrent2.add(self.drop)

    def hybrid_forward(self, F, bert_embedding, mask):
        bert_embedding = F.squeeze(bert_embedding)
        return self.from_embedding(F, bert_embedding, mask)


class DANTextClassifier(HybridBlock):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    dense_units : list[int], default = [100,100]
        Dense units for each layer after pooled embedding
    activation : string, default = relu
        Activation function for fully connected layers
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=4, dr=0.2, dense_units=[100,100]):
        super(DANTextClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim=emb_input_dim, output_dim=emb_output_dim)
        self.mlp = nn.HybridSequential()

        with self.name_scope():
            self.mlp.add(nn.AvgPool1D())
            self.mlp.add(nn.Dropout(rate=dr))
            for i in dense_units:
                self.mlp.add(nn.Dense(i, activation="relu"))
            self.mlp.add(nn.Dense(num_classes))

    def from_embedding(self, F, embedded, mask):
        """Implements the rest of the computation from the embedding
        """
        # masking the padding data
        reshape = mx.nd.reshape(mask, shape=(mask.shape[0], mask.shape[1], 1) )
        result = mx.nd.broadcast_mul(embedded, reshape)
        result = self.mlp(result)
        return result

    def hybrid_forward(self, F, data, mask):
        embedded = self.embedding(data)
        return self.from_embedding(F, embedded, mask)


class BertDANTextClassifier(DANTextClassifier):

    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=4, dr=0.2, dense_units=[100,100]):
        super(DANTextClassifier, self).__init__()
        self.mlp = nn.HybridSequential()

        with self.name_scope():
            self.mlp.add(nn.AvgPool1D())
            self.mlp.add(nn.Dropout(rate=dr))
            for i in dense_units:
                self.mlp.add(nn.Dense(i, activation="relu"))
            self.mlp.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, bert_embedding, mask):
        bert_embedding = F.squeeze(bert_embedding)
        return self.from_embedding(F, bert_embedding, mask)


class LSTMTextClassifier(HybridBlock):
    """
    Parameters
    ----------
    emb_input_dim : int
        The dimensionality of the vocabulary (input to Embedding layer)
    emb_output_dim : int
        The dimensionality of the embedding
    num_classes : int, default 2
        Number of categories in classifier output    
    dr : float, default 0.2
        Dropout rate
    """
    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=2, dr=0.2):
        super(LSTMTextClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim=emb_input_dim, output_dim=emb_output_dim)
        self.mlp = nn.HybridSequential()
        self.lstm = gluon.rnn.LSTM(100)
        self.drop = nn.Dropout(dr)
        self.out = nn.Dense(num_classes)

        with self.name_scope():
            self.mlp.add(self.drop)
            self.mlp.add(self.out)
            
    def from_embedding(self, F, embedded, mask):
        """
        Implements the rest of the computation from the embedding
        """
        # apply mask to input data
        reshape = mx.nd.reshape(mask, shape=(mask.shape[0], mask.shape[1], 1) )
        reshape = np.swapaxes(reshape, 0, 1)
        embedded = mx.nd.broadcast_mul(embedded, reshape)

        result = self.lstm(embedded)
        result = np.swapaxes(result, 0, 1)
        result = self.mlp(result)
        return result

    def hybrid_forward(self, F, data, mask):
        data = np.swapaxes(data, 0, 1)
        embedded = self.embedding(data)
        return self.from_embedding(F, embedded, mask)


class BertLSTMTextClassifier(LSTMTextClassifier):

    def __init__(self, emb_input_dim=0, emb_output_dim=0, num_classes=2, dr=0.2):
        super(LSTMTextClassifier, self).__init__()
        self.mlp = nn.HybridSequential()
        self.lstm = gluon.rnn.LSTM(60)
        self.drop = nn.Dropout(dr)
        self.out = nn.Dense(num_classes)

        with self.name_scope():
            self.mlp.add(self.drop)
            self.mlp.add(self.out)

    def hybrid_forward(self, F, bert_embedding, mask):
        bert_embedding = F.squeeze(bert_embedding)
        bert_embedding = np.swapaxes(bert_embedding, 0, 1)
        return self.from_embedding(F, bert_embedding, mask)
    
    
    
