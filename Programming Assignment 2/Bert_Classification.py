# codeing: utf-8
import io
import random
import logging
import argparse
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from typing import List, Tuple
from mxnet.gluon.data import DataLoader
from eval import eval
import gluonnlp as nlp
from utils import logging_config
from gluonnlp.model import get_model
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from gluonnlp.data.dataset import SimpleDataset
from sklearn.random_projection import SparseRandomProjection
import warnings
warnings.filterwarnings("ignore")

# Define Arguments
parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
parser.add_argument('--train_file', default="/Users/wanyuexiao/Downloads/A2/train.tsv", type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
# "/Users/wanyuexiao/Downloads/A2/semevalTest_with_keys.tsv"
parser.add_argument('--epochs', type=int, default=50, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=36)
parser.add_argument('--drop_rate', type=float, help='Dropout ratio', default=0.1)
parser.add_argument('--max_length', type=int, help='max_length', default=86)
parser.add_argument('--emb_size', type=int, help='emb_size', default=128)
parser.add_argument('--adam_epsilon', type=float, help='adam_epsilon', default=1e-8)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d', help='Pre-trained embedding source name')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')
args = parser.parse_args()

relation_types = ["Component-Whole",
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
                  "Other"]


# Function that is used to find the index of overlapping elements from two lists
def search_index(parent, child):
    if not child:
        return
    # just optimization
    lengthneedle = len(child)
    firstneedle = child[0]
    for idx, item in enumerate(parent):
        if item == firstneedle:
            if parent[idx:idx+lengthneedle] == child:
                yield tuple(range(idx, idx+lengthneedle))


# Class that is used to transform data instances
class BERTDatasetTransform(object):
    def __init__(self, tokenizer, bert_model, max_seq_length, class_labels=None, emb_dim=64, pad=True):
        self.class_labels = class_labels
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._label_map = {}
        for (i, label) in enumerate(class_labels):
            self._label_map[label] = i
        self._bert_xform = BERTSentenceTransform(tokenizer, max_seq_length, pad=0, pair=False)
        self._rand_projection = SparseRandomProjection(n_components = emb_dim)
        self._rand_projection.fit(np.random.randn(1,768)) ## initialize random projector

    def __call__(self, line):
        label, ind1, ind2, txt = line
        ind1 = int(ind1)
        ind2 = int(ind2)
        ori_text = txt.split(" ")
        ent1 = ori_text[ind1]
        ent2 = ori_text[ind2]

        label = self._label_map[label]
        result = self._bert_xform((txt,))
        result_idx = result[0].tolist()
        # Tokenize the ent1 and ent2 to get corresponding sub-world index
        ent1 = self._bert_xform((ent1,))[0][1:-1].tolist()
        ent2 = self._bert_xform((ent2,))[0][1:-1].tolist()
        # Feeding the results of BERTSentenceTransform into the Pre-trained Bert model
        words, valid_length, segments = mx.nd.array([result[0]]), mx.nd.array([result[1]]), mx.nd.array([result[2]])
        print(words.shape, valid_length.shape, segments.shape)
        seq_encoding, cls_encoding = self.bert_model(words, segments, valid_length) # [1, 768]
        print(seq_encoding.shape, cls_encoding.shape)

        # Using the search_index function to find the potential index of entity based on the sentence representation
        try:
            ent1_idx = list(search_index(result_idx, ent1))[0]
            ent1_ = np.mean(seq_encoding[:, list(ent1_idx), :].asnumpy(), axis=1, keepdims=True).squeeze(0)
        except IndexError:
            ent1_idx = ind1 + 1
            ent1_ = np.mean(seq_encoding[:, ent1_idx, :].asnumpy(), axis=1, keepdims=True).squeeze(0)
        try:
            ent2_idx = list(search_index(result_idx, ent2))[0]
            ent2_ = np.mean(seq_encoding[:, list(ent2_idx), :].asnumpy(), axis=1, keepdims=True).squeeze(0)
        except IndexError:
            ent2_idx = ind2 + 1
            ent2_ = np.mean(seq_encoding[:, ent2_idx, :].asnumpy(), axis=1, keepdims=True).squeeze(0)

        print(result_idx, ent1, ent1_idx)
        print()

        cls_encoding = cls_encoding.asnumpy()
        cls_projection = mx.nd.array(self._rand_projection.transform(cls_encoding)) # [1,128]
        ent1_ = mx.nd.array(self._rand_projection.transform(ent1_)) # [1,128]
        ent2_ = mx.nd.array(self._rand_projection.transform(ent2_))
        return cls_projection, ent1_, ent2_, valid_length, mx.nd.array([label]) #(1, 128) (1, 128) (1, 128) (1,) (1,)


# Function that is used to convert tsv file into array-like object
def load_tsv_to_array(file) -> List:
    """
        Inputs: file path
        Outputs: list/array of 3-tuples, each representing a data instance
    """
    arr = []
    with io.open(file, 'r') as fp:
        for line in fp:
            els = line.split('\t')
            els[3] = els[3].strip()
            els[2] = int(els[2])
            els[1] = int(els[1])
            arr.append(tuple(els))
    return arr


# Function that is used to generate datasets
def load(train_file, val_file=None, test_file=None) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    print("start loading data and save it as a array")
    train_array = load_tsv_to_array(train_file)

    if val_file:
        val_array = load_tsv_to_array(val_file)
    else:
        random.shuffle(train_array)
        train_array, val_array = np.split(train_array, [int(len(train_array)*0.8)])
        train_array = train_array.tolist()
        val_array = val_array.tolist()
        print("update length of train and val ", len(train_array), len(val_array))

    if test_file:
        test_array = load_tsv_to_array(test_file)
    else:
        test_array = None

    print("finish data loading")
    return train_array, val_array, test_array


# Function that is used to convert arrays to dataloader object
def process_dataset(transform, labels, train_file, val_file, test_file, batch_size, max_len, train_mode=True):
    train_dataset, val_dataset, test_dataset = load(train_file, val_file, test_file)
    train_dataset = SimpleDataset(list(map(transform, train_dataset)))
    val_dataset = SimpleDataset(list(map(transform, val_dataset)))
    if test_dataset:
        test_dataset = SimpleDataset(list(map(transform, test_dataset)))
    else:
        test_dataset = None

    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(), nlp.data.batchify.Stack('float32'))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=train_mode, batchify_fn=batchify_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, batchify_fn=batchify_fn)
    if test_dataset:
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, batchify_fn=batchify_fn)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


class FCLayer(nn.Block):
    def __init__(self, output_dim, dropout_rate=0.1, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Dense(output_dim)
        self.tanh = nn.Activation('tanh')

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BERTClassifier(nn.Block):
    def __init__(self, num_classes, dropout_rate):
        super(BERTClassifier, self).__init__()
        self.num_classes = num_classes
        self.drop_rate = dropout_rate
        self.net = nn.Sequential()
        self.net.add(FCLayer(256, dropout_rate=self.drop_rate))

        self.output = nn.HybridSequential()
        with self.output.name_scope():
            self.output.add(nn.Dense(units=256, flatten=True, activation='relu'))
            self.output.add(nn.Dense(units=256, activation='relu'))
            self.output.add(nn.Dense(self.num_classes, use_bias=True))

    # def forward(self, inputs, segment_types, seq_len):
    def forward(self, cls, ent1, ent2, valid_len):
        cls_result = self.net(cls)
        ent1_result = self.net(ent1)
        ent2_result = self.net(ent2)
        concat = mx.nd.concat(cls_result, ent1_result, ent2_result, dim=1)
        # concat = mx.nd.concat(cls, ent1, ent2, dim=2)
        out = self.output(concat)
        return out


def train_classifier(model, train_loader, val_loader, test_loader, epoch, lr, loss_fn, ctx=mx.cpu()):
    differentiable_params = []
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
    num_classes = 19

    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    for epoch in range(epoch):
        epoch_loss = 0
        for i, x in enumerate(train_loader):
            cls, ent1, ent2, valid_len, label = x
            cls = cls.as_in_context(ctx)
            ent1 = ent1.as_in_context(ctx)
            ent2 = ent2.as_in_context(ctx)
            valid_len = valid_len.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output = model(cls, ent1, ent2, valid_len)
                l = loss_fn(output, label).mean()
            l.backward()
            trainer.step(1)
            epoch_loss += l.asscalar()
        logging.info("Epoch {} loss = {}".format(epoch+1, epoch_loss))

        tr_ap, tr_acc, y_true, y_pred = evaluate(model, train_loader, num_classes)
        tr_precision, tr_recall, tr_ap = eval(y_true, y_pred, num_classes)
        logging.info("TRAINING AP = {} Acc = {} Precision = {} Recall = {}".format(tr_ap, tr_acc, tr_precision, tr_recall))

        val_ap, val_acc, y_true, y_pred = evaluate(model, val_loader, num_classes)
        val_precision, val_recall, val_ap = eval(y_true, y_pred, num_classes)
        logging.info("VALIDATION AP = {} Acc = {} Precision = {} Recall = {}".format(val_ap, val_acc, val_precision, val_recall))

    if test_loader is not None:
        logging.info("***** Training complete. *****")
        tst_ap, tst_acc, y_true, y_pred = evaluate(model, test_loader, 19)
        tst_precision, tst_recall, tst_ap = eval(y_true, y_pred, num_classes)
        logging.info("TEST AP = {} Acc = {} Precision = {} Recall = {}".format(tst_ap, tst_acc, tst_precision, tst_recall))


def evaluate(model, dataloader, num_classes, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, etc.)
    """
    acc = 0
    total_correct = 0
    total = 0

    y_true = np.array([])
    y_pred = np.array([])

    for i, x in enumerate(dataloader):
        cls, ent1, ent2, valid_len, label = x
        cls = cls.as_in_context(ctx)
        ent1 = ent1.as_in_context(ctx)
        ent2 = ent2.as_in_context(ctx)
        valid_len = valid_len.as_in_context(ctx)
        label = label.as_in_context(ctx)

        out = model(cls, ent1, ent2, valid_len)
        out_argmax = mx.nd.argmax(out, axis=1).astype('int32')
        for j in range(out.shape[0]):
            probs = mx.nd.softmax(out[j]).asnumpy()
            lab = int(label[j].asscalar())
            best_probs = np.argmax(probs)
            if lab == best_probs:
                total_correct += 1
            total += 1
        y_true = np.append(y_true, label.squeeze().asnumpy())
        y_pred = np.append(y_pred, out_argmax.asnumpy())
    acc = total_correct / float(total)
    return 0.0, acc, y_true, y_pred


if __name__ == '__main__':
    logging_config(args.log_dir, 'train', level=logging.INFO)
    ctx = mx.cpu()
    # print("file dirs ", args.train_file, args.val_file, args.test_file)

    # loading bert model
    bert, bert_vocabulary = get_model(name='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', pretrained=True, ctx=mx.cpu(), use_pooler=True, use_decoder=False, use_classifier=False)
    bert_tokenizer = nlp.data.BERTTokenizer(bert_vocabulary, lower=True)
    print('Index for [CLS] = ', bert_vocabulary['[CLS]']) # 2

    print("start bert embedding")
    transform = BERTDatasetTransform(bert_tokenizer, bert, args.max_length, class_labels=relation_types, emb_dim=args.emb_size)
    train_loader, val_loader, test_loader = process_dataset(transform=transform, labels=relation_types, train_file=args.train_file, val_file=args.val_file,
                                                            test_file=args.test_file, batch_size=args.batch_size, max_len=args.max_length, train_mode=True)

    loss_fn = gluon.loss.SoftmaxCELoss()
    net = BERTClassifier(len(relation_types), dropout_rate=args.drop_rate)
    net.initialize(ctx=ctx)
    net.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)  ## initialize model parameters on the context ctx

    print("start training ")
    train_classifier(net, train_loader, val_loader, test_loader, args.epochs, args.lr, loss_fn, ctx)
