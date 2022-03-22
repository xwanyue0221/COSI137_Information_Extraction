# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
import gluonnlp as nlp
import matplotlib.pyplot as plt

from load_data import get_data_loaders, get_data_loaders_bert
from model import CNNTextClassifier, LSTMTextClassifier, DANTextClassifier
from model import BertCNNTextClassifier, BertLSTMTextClassifier, BertDANTextClassifier
from utils import logging_config
from eval import eval


parser = argparse.ArgumentParser(description='Train a (short) text classifier - via convolutional or other standard architecture')
parser.add_argument('--train_file', default=None, type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', default=None, type=str, help='File containing file representing the input VALIDATION data')
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=50)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--seq_length', type=int, help='Max sequence length', default = 64)
parser.add_argument('--embedding_source', type=str, default='glove.6B.100d', help='Pre-trained embedding source name (GluonNLP)')
parser.add_argument('--lstm', action='store_true', help='Use an LSTM layer instead of CNN encoder')
parser.add_argument('--dan', action='store_true', help='Use a DAN enocder instead of CNN encoder')
parser.add_argument('--dense_dan_layers', type=str, default='100,100', help='List if integer dense unit layers (DAN only)')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')
parser.add_argument('--embedding_size', type=int, default=300, help='Embedding size (if random) (DEFAULT = 300)')
parser.add_argument('--filter_sizes', type=str, default='3,4', help='List of integer filter sizes (for CNN only)')
parser.add_argument('--min_length', type=int, default=64, help='Minimum size of sequence (possibly needed for CNN)')
parser.add_argument('--num_filters', type=int, default=100, help='Number of filters (of each size)')
parser.add_argument('--num_conv_layers', type=int, default=3, help='Number of convolutional/pool layers')
parser.add_argument('--pad_data', action='store_true', help='Explicitly pad all data to seq_length', default=False)
parser.add_argument('--use_bert_embeddings', action='store_true', help='Use Bert as a feature extractor (no fine-tuning)', default=False)
parser.add_argument('--bert_cache_dir', type=str, help='Directory to store bert embeddings', default="./")
parser.add_argument('--bert_proj_dim', type=int, help='Down-projected dimensions from BERT', default=128)

args = parser.parse_args()
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


def get_model(num_classes, vocab_size=0, embedding_size=0):

    filters = [ int(x) for x in args.filter_sizes.split(',') ]
    if args.use_bert_embeddings:
        if args.lstm:
            model = BertLSTMTextClassifier(emb_output_dim=embedding_size, num_classes=num_classes, dr=args.dropout)
        elif args.dan:
            print("start dan")
            model = BertDANTextClassifier(emb_output_dim=embedding_size, num_classes=num_classes, dr=args.dropout)
        else:
            model = BertCNNTextClassifier(emb_output_dim=embedding_size, filter_widths=filters,
                                          # num_conv_layers=args.num_conv_layers,
                                          num_filters=args.num_filters,
                                          num_classes=num_classes, dr=args.dropout)
        ## initialize model
        model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)  
    else:
        emb_input_dim, emb_output_dim = vocab_size, embedding_size
        if args.lstm:
            model = LSTMTextClassifier(emb_input_dim=emb_input_dim, emb_output_dim=emb_output_dim,
                                       num_classes=num_classes, dr=args.dropout)
        elif args.dan:
            dense_units = [ int(x) for x in args.dense_dan_layers.split(',') ]
            model = DANTextClassifier(emb_input_dim=emb_input_dim,
                                      emb_output_dim=emb_output_dim,
                                      num_classes=num_classes, dr=args.dropout, dense_units=dense_units)
        else:
            model = CNNTextClassifier(emb_input_dim=emb_input_dim, emb_output_dim=emb_output_dim,
                                      filter_widths=filters, num_classes=num_classes,
                                      dr=args.dropout, num_conv_layers=args.num_conv_layers,
                                      num_filters=args.num_filters)
        # initialize model parameters on the context ctx
        model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)  
        if not args.random_embedding:
            # set the embedding layer parameters to pre-trained embedding
            model.embedding.weight.set_data(vocab.embedding.idx_to_vec) 
        elif args.fixed_embedding:
            model.embedding.collect_params().setattr('grad_req', 'null')
    return model


def train_classifier(model, train_loader, val_loader, test_loader, num_classes, ctx=mx.cpu()):

    # model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    start_ap, start_acc, start_y_true, start_y_pred = evaluate(model, val_loader, num_classes)
    logging.info("Starting AP = {} Acc = {}".format(start_ap, start_acc))
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, (ids, lens, mask, label) in enumerate(train_loader):
            ids = ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            mask = mask.as_in_context(ctx)
            with autograd.record():
                output = model(ids, mask)
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
        # if epoch == (args.epochs-1):

    if test_loader is not None:
        tst_ap, tst_acc = evaluate(model, test_loader, num_classes)
        logging.info("***** Training complete. *****")
        logging.info("TEST AP = {} Acc = {}".format(tst_ap, tst_acc))


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

    for i, (ids, lens, mask, label) in enumerate(dataloader):
        out = model(ids, mask)
        out_argmax = mx.nd.argmax(out,axis=1)
        # print("predicted_out ", out_argmax)
        total_correct += (out_argmax == label.squeeze()).sum().asscalar()
        total += label.shape[0]
        y_true = np.append(y_true, label.squeeze().asnumpy())
        y_pred = np.append(y_pred, out_argmax.asnumpy())
    acc = total_correct / float(total)

    # class_labels = ["Objectives", "Outcome", "Prior", "Approach"]
    # print(classification_report(y_label, y_predict, target_names=class_labels, zero_division=1))
    return 0.0, acc, y_true, y_pred

if __name__ == '__main__':
    logging_config(args.log_dir, 'train', level=logging.INFO)
    class_labels = ["Objectives", "Outcome", "Prior", "Approach"]
    ctx = mx.cpu() ## or mx.gpu(N) if GPU device N is available
    
    if args.use_bert_embeddings:
        train_loader, val_loader, test_loader = \
            get_data_loaders_bert(class_labels,
                                  args.train_file,
                                  args.val_file,
                                  args.test_file,
                                  args.batch_size,
                                  args.seq_length,
                                  args.min_length,
                                  False,
                                  "sentence",
                                  "label0",
                                  cache_dir=args.bert_cache_dir,
                                  emb_size=args.bert_proj_dim)
        model = get_model(len(class_labels), embedding_size=args.bert_proj_dim)
    else:
        vocab, train_loader, val_loader, test_loader = \
            get_data_loaders(class_labels,
                             args.train_file,
                             args.val_file,
                             args.test_file,
                             args.batch_size,
                             args.seq_length,
                             args.min_length,
                             False,
                             "sentence",
                             "label0")
        ## setup embeddings
        if args.embedding_source and not args.random_embedding:
            if ':' in args.embedding_source:
                typ, src = tuple(args.embedding_source.split(':'))
                embedding = nlp.embedding.create(typ, source=src)
            else:
                embedding = nlp.embedding.create('glove', source=args.embedding_source)
            vocab.set_embedding(embedding)
            _, emb_size = vocab.embedding.idx_to_vec.shape
            # set embeddings to random for out of vocab items
            oov_items = 0
            for word in vocab.embedding._idx_to_token:
                if (vocab.embedding[word] == mx.nd.zeros(emb_size)).sum() == emb_size:
                    oov_items += 1
                    vocab.embedding[word] = mx.nd.random.normal(0.0, 0.1, emb_size)
            logging.info("** There are {} out of vocab items **".format(oov_items))
        else:
            emb_size = args.embedding_size
        model = get_model(len(class_labels), vocab_size=len(vocab), embedding_size=emb_size)
    
    train_classifier(model, train_loader, val_loader, test_loader, len(class_labels), ctx)
