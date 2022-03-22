# coding: utf-8

import re
import io
import io
import os
import time
import pickle
import argparse
import random
import logging
import warnings
import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block, nn
import gluonnlp as nlp
from gluonnlp.model import get_model
from sklearn.random_projection import SparseRandomProjection
from gluonnlp.data.dataset import SimpleDataset
import json
import collections

from gluonnlp.data import BERTTokenizer
from gluonnlp.data import BERTSentenceTransform


class JsonlDataset(SimpleDataset):
    """A dataset wrapping over a jsonlines (.jsonl) file, each line is a json object.

    Parameters:
        filename (str): Path to the .jsonl file.
        text_key (str): String corresponding to text key
        label_keky (str): String corresponding to label key
        encoding (str): File encoding format. (default 'utf8')
    """
    def __init__(self, filename, text_key, label_key, encoding='utf8'):
        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._text_key = text_key
        self._label_key = label_key

        super(JsonlDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            samples = []
            with open(filename, 'r', encoding=self._encoding) as fin:
                for line in fin.readlines():
                    samples.append(json.loads(line, object_pairs_hook=collections.OrderedDict))
            samples = self._read_samples(samples)
            all_samples += samples
        return all_samples

    def _read_samples(self, samples):
        m_samples = []
        for s in samples:
            m_samples.append((s[self._text_key], s[self._label_key]))
        return m_samples

    
class TextTransform(object):

    def __init__(self,
                 tokenizer,
                 vocabulary,
                 pad=True,
                 max_seq_len=128,
                 min_seq_len=0):
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary
        self._max_seq_len = max_seq_len
        self._min_seq_len = min_seq_len        
        self._pad = pad

    def __call__(self, txt):
        tokens = self._tokenizer(txt)
        ids = self._vocabulary[tokens][:self._max_seq_len]
        orig_len = len(ids)
        ones = np.ones(orig_len, dtype='float32')
        if self._pad:
            zeros = np.zeros(self._max_seq_len - orig_len, dtype='float32')
            mask = np.concatenate([ones, zeros])
            ids = ids + [0] * (self._max_seq_len - orig_len)
        elif self._min_seq_len > 0:
            new_len = max(self._min_seq_len, orig_len)
            if new_len > 0:
                zeros = np.zeros(new_len - orig_len, dtype='float32')
                mask = np.concatenate([ones, zeros])
                ids = ids + [0] * (new_len - orig_len)
        else:
            mask = ones
        ids = np.array(ids, dtype='float32')
        orig_len = np.array([orig_len], dtype='float32')
        return ids, orig_len, mask


class ClassifierTransform(object):
    """
    Parameters:
        tokenizer (obj): Callable object that splits a string into a list of tokens
        vocabulary (:class:`gluonnlp.data.Vocab`): GluonNLP vocabulary
        max_seq_len (int): Maximum sequence/text length
        pad (bool): Whether to pad data to maximum length
        class_labels (list): List of strings for the class labels
    """

    def __init__(self,
                 tokenizer,
                 vocabulary,
                 max_seq_len,
                 min_seq_len,
                 pad=True,
                 class_labels=None):
        self._text_xform = TextTransform(tokenizer, vocabulary, pad, max_seq_len, min_seq_len)
        self._class_labels = class_labels
        self._label_map = {}
        for (i, label) in enumerate(class_labels):
            self._label_map[label] = i

    def __call__(self, labeled_text):
        """
        Parameters:
            labeled_text: tuple of str
                Input instances of (text, label) pairs
        Returns:
            np.array: token ids, shape (seq_length,)
            np.array: valid length, shape (1,)
            np.array: mask, shape (seq_length,)
            np.array: label id, shape (1,)
        """
        text, label = labeled_text
        input_ids, valid_length, mask = self._text_xform(text)
        label = self._label_map[label]
        label = np.array([label], dtype='float32')
        return input_ids, valid_length, mask, label


def build_vocabulary(dataset, tokenizer):
    """
    Parameters:
        dataset (:class:`JsonlDataset`): JsonlDataset to build vocab over
        tokenizer (obj): Callable object to split strings into tokens
    Returns:
        (:class:`gluonnlp.data.Vocab`): GluonNLP vocab object
    """
    all_tokens = []
    for (txt, label)  in dataset:
        all_tokens.extend(tokenizer(txt))
    counter = nlp.data.count_tokens(all_tokens)
    vocab = nlp.Vocab(counter)
    return vocab


class BasicTokenizer(object):
    """Callable object to split strings into lists of tokens
    """
    def __call__(self, txt):
        return txt.split(' ')


def get_data_loaders(class_labels, train_file, dev_file, test_file, batch_size, max_len, min_len, pad, text_key, label_key):
    """
    Parameters:
        class_labels (list): List of strings representing class labels
        train_file (str): Path to training data in Jsonl format
        dev_file (str): Path to dev data in Jsonl format
        test_file (str): Path to test data in Jsonl format
        batch_size (int): Batch size
        max_len (int): Maximum sequence length
        pad (bool): Flag for whether to pad data to max_len
        text_key (str): Json attribute key corresponding to text data
        label_key (str): Json attribute key corresponding to label
    Returns:
        (tuple): Tuple of:
            :class:`gluonnlp.data.Vocab`
            :class:`mx.gluon.data.DataLoader`
            :class:`mx.gluon.data.DataLoader`
            :class:`mx.gluon.data.DataLoader`
    """
    tokenizer = BasicTokenizer()
    train_ds = JsonlDataset(train_file, text_key=text_key, label_key=label_key)
    vocabulary = build_vocabulary(train_ds, tokenizer)
    transform = ClassifierTransform(tokenizer, vocabulary, max_len, min_len, pad=pad, class_labels=class_labels)
    data_train = mx.gluon.data.SimpleDataset(list(map(transform, train_ds)))
    data_train_lengths = data_train.transform(
        lambda ids, lengths, mask, label_id: lengths, lazy = False)
    # bucket sampler for training
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                                          nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack())
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_lengths,
        batch_size=batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    loader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=4,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)

    dev_ds = JsonlDataset(dev_file, text_key=text_key, label_key=label_key)
    data_dev = mx.gluon.data.SimpleDataset(list(map(transform, dev_ds)))
    loader_dev = mx.gluon.data.DataLoader(
        data_dev,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        batchify_fn=batchify_fn)

    if test_file:
        test_ds = JsonlDataset(test_file, text_key=text_key, label_key=label_key)
        data_test = mx.gluon.data.SimpleDataset(list(map(transform, test_ds)))
        loader_test = mx.gluon.data.DataLoader(
            data_test,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            batchify_fn=batchify_fn)
    else:
        loader_test = None
        
    return vocabulary, loader_train, loader_dev, loader_test


class BERTDatasetTransform(object):
    """Dataset transformation for BERT-style sentence classification or regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    bert_model: BERT
        Bert encoder
    max_seq_length : int.
        Maximum sequence length of the sentences.
    labels : list of int , float or None. defaults None
        List of all label ids for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    emb_dim: int, default 64
        Use a random projection to compress BERT encodings
    """
    def __init__(self, tokenizer, bert_model, max_seq_length, class_labels=None, emb_dim=64, pad=True):
        self.class_labels = class_labels
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length
        self._label_map = {}
        for (i, label) in enumerate(class_labels):
            self._label_map[label] = i
        self._bert_xform = BERTSentenceTransform(tokenizer, max_seq_length, pad=pad, pair=False)
        self._rand_projection = SparseRandomProjection(n_components = emb_dim)
        self._rand_projection.fit(np.random.randn(1,768)) ## initialize random projector
        

    def __call__(self, line):
        """
        Parameters:
            line tuple of str: A tuple of (text, label)
        Return:
             4-tuple
        """
        ## implement the transform here
        ## 1) get the label index from _label_map        
        ## 2) Use self._bert_xform to process the text, note the input MUST be a tuple with a single element - i.e. (text,)
        ##    This will result in input_ids, valid_length and segment_ids (see https://nlp.gluon.ai/model_zoo/bert/index.html)
        ## 3) Run the resulting tokens/ids/lengths through self.bert_model (again, see https://nlp.gluon.ai/model_zoo/bert/index.html)
        ##    This will result int the sequence encoding (and "CLS" encoding, which can be ignored)    
        ## 4) To save space, use a random projection self._rand_projection to reduce the dimensionality of the BERT encodings
        ## 5) To conform to our model api, the method should return:
        ##         seq_encoding - the encoding (after projection)
        ##         valid_length - an ND array with a single integer (with the valid length for the sequence)
        ##         mask         - the mask isn't used here so should just be an array of ones with size equal to max_seq_length
        ##         label        - an ND array with a single integer (the label index for this example)

        text, label = line
        label = self._label_map[label]
        result = self._bert_xform((text,))
        words, valid_length, segments = mx.nd.array([result[0]]), mx.nd.array([result[1]]), mx.nd.array([result[2]])
        ones = mx.nd.ones(self.max_seq_length, dtype='float32')

        seq_encoding, cls_encoding = self.bert_model(words, segments, valid_length)
        seq_encoding = seq_encoding.squeeze(0).asnumpy()
        seq_projection = mx.nd.array(self._rand_projection.transform(seq_encoding))
        # seq_projection = mx.nd.array(self._rand_projection.fit_transform(seq_encoding))
        seq_projection = seq_projection.expand_dims(0)
        return seq_projection, valid_length, ones, mx.nd.array([label])


def process_dataset(transform, class_labels, data_file, batch_size, max_len, train_mode=True,
                    text_key="sentence", label_key="label0", cache_file=None):
    if cache_file and os.path.exists(cache_file):
        with io.open(cache_file, 'rb') as istr:
            data_ds = pickle.load(istr)
    else:
        dataset = JsonlDataset(data_file, text_key=text_key, label_key=label_key)
        data_ds = mx.gluon.data.SimpleDataset(list(map(transform, dataset)))
        if cache_file:
            with io.open(cache_file, 'wb') as ostr:
                pickle.dump(data_ds, ostr)
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(), nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack('float32'))
    loader = gluon.data.DataLoader(dataset=data_ds, batch_size=batch_size, shuffle=train_mode, batchify_fn=batchify_fn)
    return loader


def get_data_loaders_bert(class_labels, train_file, dev_file, test_file, batch_size, max_len, min_len,
                          pad, text_key, label_key, emb_size=64, cache_dir=None):
    bert, bert_vocabulary = get_model(
        name='bert_12_768_12',
        dataset_name='book_corpus_wiki_en_uncased',
        pretrained=True,
        ctx=mx.cpu(),
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    do_lower_case = True
    bert_tokenizer = BERTTokenizer(bert_vocabulary, lower=do_lower_case)

    print("start bert embedding")
    transform = BERTDatasetTransform(bert_tokenizer, bert, max_len, class_labels=class_labels, emb_dim=emb_size)
    if cache_dir and not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    train_loader = process_dataset(transform, class_labels, train_file, batch_size, max_len, train_mode=True, cache_file = os.path.join(cache_dir, 'train.pkl') if cache_dir else None)
    if dev_file or os.path.exists(os.path.join(cache_dir, 'dev.pkl')):
        dev_loader = process_dataset(transform,
                                     class_labels,
                                     dev_file,
                                     batch_size,
                                     max_len,
                                     train_mode=False,
                                     cache_file = os.path.join(cache_dir, 'dev.pkl') if cache_dir else None)
    else:
        dev_loader = None
    if test_file or os.path.exists(os.path.join(cache_dir, 'test.pkl')):        
        test_loader = process_dataset(transform,
                                      class_labels,
                                      test_file,
                                      batch_size,
                                      max_len,
                                      train_mode=False,
                                      cache_file= os.path.join(cache_dir, 'test.pkl') if cache_dir else None)
    else:
        test_loader = None
    return train_loader, dev_loader, test_loader

        
