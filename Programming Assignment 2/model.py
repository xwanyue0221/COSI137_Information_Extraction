# codeing: utf-8

import math
import os
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.gluon.loss import L2Loss

from mxnet.gluon.block import HybridBlock, Block
from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, MultiHeadAttentionCell


def _get_attention_cell(attention_cell, units=None, scaled=True, num_heads=None, use_bias=False, dropout=0.0):
    """
    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell), \
            'attention_cell must be either string or AttentionCell. Received attention_cell={}' \
                .format(attention_cell)
        return attention_cell


def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


class RelationClassifier(HybridBlock):
    def __init__(self, emb_input_dim, emb_output_dim, max_seq_len=32, num_classes=19, prefix=None, params=None):
        super(RelationClassifier, self).__init__(prefix=prefix, params=params)

        self.seq_length = max_seq_len
        self.emb_dim = emb_output_dim
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.attn_layer = BaseEncoder(units=emb_output_dim, max_length=max_seq_len)
            self.binrel_encoder = BinRelEncoder(units=128, max_length=max_seq_len)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(nn.Dense(units=256, flatten=True, activation='relu'))
                self.output.add(nn.Dense(units=256, activation='relu'))
                self.output.add(gluon.nn.Dense(units=num_classes, use_bias=True))


    def hybrid_forward(self, F, data, inds):
        embedded = self.embedding(data) ## shape (batch_size, length, emb_dim)
        attended = self.attn_layer(embedded) ## shape (batch_size, length, emb_output_dim)
        encoded = self.binrel_encoder(attended, inds) ## shape (batch_size, 2, attention_out)
        return self.output(encoded)


class PositionwiseFFN(HybridBlock):
    """
    Taken from the gluon-nlp library.
    """
    def __init__(self, units=512, hidden_size=1024, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False, activation=activation, weight_initializer=weight_initializer, bias_initializer=bias_initializer, prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False, weight_initializer=weight_initializer, bias_initializer=bias_initializer, prefix='ffn_2_')
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        outputs = self.ffn_1(inputs)
        outputs = self.ffn_2(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


class BaseEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """

    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BaseEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        arg_inputs: Symbol or NDArray
            Input arguments. Shape (batch_size, 2)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the encoder cell. Shape (batch_size, 2, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights = self.attention_cell(inputs, inputs, None, None)
        outputs = self.proj(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        return outputs


class BaseEncoder(HybridBlock):

    def __init__(self, attention_cell='multi_head',
                 units=512, hidden_size=2048, max_length=64,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BaseEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, \
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
                .format(units, num_heads)
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            ## !!! Original code creates a number of attention layers
            ## !!! Hard-coded here for a single base encoder cell for simplicity
            # self.transformer_cells = nn.HybridSequential()
            # for i in range(num_layers):
            #    self.transformer_cells.add(
            self.base_cell = BaseEncoderCell(
                units=units,
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_cell=attention_cell,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                dropout=dropout,
                use_residual=use_residual,
                scaled=scaled,
                output_attention=output_attention,
                prefix='transformer')

    def __call__(self, inputs):  # pylint: disable=arguments-differ
        return super(BaseEncoder, self).__call__(inputs)

    def hybrid_forward(self, F, inputs, position_weight):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        arg_pos: int array pair

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        """
        batch_size = inputs.shape[0]
        steps = F.arange(self._max_length)
        inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                   self._max_length,
                                                                   self._units), axis=0))
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = self.base_cell(inputs)
        return outputs


class BinRelEncoderCell(HybridBlock):
    """This is a TransformerEncoder block/layer that generates outputs corresponding to
    exactly two positions in the input.  These should be integer offsets (0-based) provided
    as an ndarray (with exactly two elements).
    """
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=128, num_heads=4, scaled=True,
                 dropout=0.2, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BinRelEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, arg_inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        arg_inputs: Symbol or NDArray
            Input arguments. Shape (batch_size, 2)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the encoder cell. Shape (batch_size, 2, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        ## the query should just be the inputs for the two RELATION ARGUMENTS (e1, e2)
        arg_outputs, attention_weights = self.attention_cell(arg_inputs, inputs, None, None)
        arg_outputs = self.proj(arg_outputs)
        arg_outputs = self.dropout_layer(arg_outputs)
        if self._use_residual:
            arg_outputs = arg_outputs + arg_inputs
        arg_outputs = self.layer_norm(arg_outputs)
        arg_outputs = self.ffn(arg_outputs)
        return arg_outputs


class BinRelEncoder(HybridBlock):
    """Same as a TransformerEncoder but generating only two hidden outputs
    at the positions of the two relation arguments.
    """
    def __init__(self, attention_cell='multi_head',
                 units=128, hidden_size=256, max_length=64,
                 num_heads=4, scaled=True, dropout=0.2,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(BinRelEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, \
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
                .format(units, num_heads)
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.binrel_cell = BinRelEncoderCell(
                units=units,
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_cell=attention_cell,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                dropout=dropout,
                use_residual=use_residual,
                scaled=scaled,
                output_attention=output_attention,
                prefix='binrel_transformer')

    def __call__(self, inputs, arg_pos): #pylint: disable=arguments-differ
        return super(BinRelEncoder, self).__call__(inputs, arg_pos)

    def hybrid_forward(self, F, inputs, arg_pos): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        arg_pos: int array pair

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        """
        batch_size = inputs.shape[0]
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        ## take/slice the inputs at the argument positions across the batch
        offsets = F.transpose(F.expand_dims(F.arange(batch_size) * self._max_length, axis=0))
        input_stacked = F.reshape(inputs, (-1, self._units))
        to_take = arg_pos + offsets
        args = F.take(input_stacked, to_take, axis=0)

        outputs = self.binrel_cell(inputs, args)
        return outputs

