# coding: utf-8

import random
import io
import re
import gluonnlp as nlp
import numpy as np
import mxnet as mx


def load_tsv_to_array(fname):
    """
    Inputs: file path
    Outputs: list/array of 3-tuples, each representing a data instance
    """
    arr = []
    with io.open(fname, 'r') as fp:
        for line in fp:
            els = line.split('\t')
            els[3] = els[3].strip()
            els[2] = int(els[2])
            els[1] = int(els[1])
            arr.append(tuple(els))
    return arr


relation_types = [
    "Component-Whole",
    "Component-Whole-Inv",
    "Instrument-Agency",
    "Instrument-Agency-Inv",
    "Member-Collection",
    "Member-Collection-Inv",
    "Cause-Effect",
    "Cause-Effect-Inv",
    "Entity-Destination",
    "Entity-Destination-Inv",
    "Content-Container",
    "Content-Container-Inv",
    "Message-Topic",
    "Message-Topic-Inv",
    "Product-Producer",
    "Product-Producer-Inv",
    "Entity-Origin",
    "Entity-Origin-Inv",
    "Other"
]


#    - Parse the input data by getting the word sequence and the argument POSITION IDs for e1 and e2
#   [[w_1, w_2, w_3, .....], [pos_1, pos_2], [label_id]]  for EACH data instance/sentence/argpair
def load_dataset(train_file, val_file=None, test_file=None, max_length=32):
    """
    Inputs: training, validation and test files in TSV format
    Outputs: vocabulary (with attached embedding), training, validation and test datasets ready for neural net training
    """
    train_array = load_tsv_to_array(train_file)
    if val_file:
        val_array = load_tsv_to_array(val_file)
    else:
        random.shuffle(train_array)
        train_array, val_array = np.split(train_array, [int(len(train_array) * 0.8)])
        train_array = train_array.tolist()
        val_array = val_array.tolist()
        print("update length of train and val ", len(train_array), len(val_array))
    if test_file:
        test_array = load_tsv_to_array(test_file)
    else:
        test_array = None

    vocabulary = build_vocabulary(train_array, val_array, test_array)
    train_dataset = preprocess_dataset(train_array, vocabulary, max_length)
    val_dataset = preprocess_dataset(val_array, vocabulary, max_length)
    if test_array:
        test_dataset = preprocess_dataset(test_array, vocabulary, max_length)
    else:
        test_dataset = None

    data_transform = BasicTransform(relation_types, max_length)
    return vocabulary, train_dataset, val_dataset, test_dataset, data_transform


def tokenize(txt):
    """
    Tokenize an input string. Something more sophisticated may help . . . 
    """
    txt = re.sub(r'[^\w\s-]', '', txt)
    return txt.lower().split(' ')


def build_vocabulary(tr_array, val_array, tst_array=None):
    """
    Inputs: arrays representing the training, validation and test data
    Outputs: vocabulary (Tokenized text as in-place modification of input arrays or returned as new arrays)
    """
    all_tokens = []
    for i, instance in enumerate(tr_array):
        label, e1, e2, text = instance
        tokens = tokenize(text)
        tr_array[i] = (label, e1, e2, tokens)  ## IN-PLACE modification of tr_array
        all_tokens.extend(tokens)

    for i, instance in enumerate(val_array):
        label, e1, e2, text = instance
        tokens = tokenize(text)
        val_array[i] = (label, e1, e2, tokens)  ## IN-PLACE modification

    if tst_array:
        for i, instance in enumerate(tst_array):
            label, e1, e2, text = instance
            tokens = tokenize(text)
            tst_array[i] = (label, e1, e2, tokens)  ## IN-PLACE modification

    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    return vocab


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    label, ind1, ind2, text_tokens = x
    data = vocab[text_tokens]  ## map tokens (strings) to unique IDs
    data = data[:max_len]  ## truncate to max_len
    return label, ind1, ind2, data


def preprocess_dataset(dataset, vocab, max_len):
    preprocessed_dataset = [_preprocess(x, vocab, max_len) for x in dataset]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 32
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """

    def __init__(self, labels, max_len=32):
        self._max_seq_length = max_len
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i

    def __call__(self, label, ind1, ind2, data):
        label_id = self._label_map[label]
        padded_data = data + [0] * (self._max_seq_length - len(data))
        inds = mx.nd.array([ind1, ind2])
        return mx.nd.array(padded_data, dtype='int32'), inds, mx.nd.array([label_id], dtype='int32')
