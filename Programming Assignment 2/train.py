# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import autograd
import gluonnlp as nlp
from eval import eval
import warnings
from load_data import load_dataset
from model import RelationClassifier
from utils import logging_config
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train a simple binary relation classifier')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data', default="/Users/wanyuexiao/Downloads/A2/train.tsv")
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=15, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.0001)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=36)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.1)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.200d', help='Pre-trained embedding source name')
parser.add_argument('--log_dir', type=str, default='.', help='Output directory for log file')
parser.add_argument('--fixed_embedding', action='store_true', help='Fix the embedding layer weights')
parser.add_argument('--random_embedding', action='store_true', help='Use random initialized embedding layer')

args = parser.parse_args()
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


def classify_test_data(vocabulary, transformer, data_test, ctx=mx.cpu()):
    """
    Generate predictions on the test data and write to file in same format as training data
    """
    raise NotImplementedError('Implement predictor')


def train_classifier(vocabulary, transformer, data_train, data_val, data_test=None, ctx=mx.cpu()):
    """
    Main loop for training a classifier
    """
    data_train = gluon.data.SimpleDataset(data_train).transform(transformer)
    data_val = gluon.data.SimpleDataset(data_val).transform(transformer)
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False)

    if data_test:
        data_test = gluon.data.SimpleDataset(data_test).transform(transformer)
        test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape if vocabulary.embedding else (len(vocabulary), 128)
    num_classes = 19  # XXX - parameterize and/or derive from dataset
    model = RelationClassifier(emb_input_dim, emb_output_dim, num_classes=num_classes)
    differentiable_params = []

    model.initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx, force_reinit=True)  ## initialize model parameters on the context ctx
    if not args.random_embedding:
        model.embedding.weight.set_data(vocab.embedding.idx_to_vec)  ## set the embedding layer parameters to pre-trained embedding
    elif args.fixed_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    # model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging
    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    for epoch in range(args.epochs):
        epoch_loss = 0
        for i, x in enumerate(train_dataloader):
            data, inds, label = x
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            inds = inds.as_in_context(ctx)
            with autograd.record():
                output = model(data, inds)
                l = loss_fn(output, label).mean()
            l.backward()
            grads = [p.grad(ctx) for p in differentiable_params]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1)  ## step = 1 since we took the mean of the loss over the batch
            epoch_loss += l.asscalar()

        logging.info("Epoch loss = {}".format(epoch_loss))
        tr_ap, tr_acc, y_true, y_pred = evaluate(model, train_dataloader, num_classes)
        tr_precision, tr_recall, tr_ap = eval(y_true, y_pred, num_classes)
        logging.info("TRAINING AP = {} Acc = {} Precision = {} Recall = {}".format(tr_ap, tr_acc, tr_precision, tr_recall))

        val_ap, val_acc, y_true, y_pred = evaluate(model, val_dataloader, num_classes)
        val_precision, val_recall, val_ap = eval(y_true, y_pred, num_classes)
        logging.info("VALIDATION AP = {} Acc = {} Precision = {} Recall = {}".format(val_ap, val_acc, val_precision, val_recall))


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

    for i, (data, inds, label) in enumerate(dataloader):
        out = model(data, inds)
        out_argmax = mx.nd.argmax(out, axis=1).astype('int32')
        for j in range(out.shape[0]):
            probs = mx.nd.softmax(out[j]).asnumpy()
            lab = int(label[j].asscalar())
            best_probs = np.argmax(probs)
            if lab == best_probs:
                total_correct += 1
            total += 1
        # total_correct += (out_argmax == label.squeeze()).sum().asscalar()
        # total += label.shape[0]
        y_true = np.append(y_true, label.squeeze().asnumpy())
        y_pred = np.append(y_pred, out_argmax.asnumpy())
    acc = total_correct / float(total)
    return 0.0, acc, y_true, y_pred

# def evaluate(model, dataloader, ctx=mx.cpu()):
#     all_scores = []
#     all_labels = []
#     for i, (data, inds, label) in enumerate(dataloader):
#         out = model(data, inds)
#         predictions = mx.nd.argmax(out, axis=1).astype('int32')
#         for j in range(out.shape[0]):
#             probs = mx.nd.softmax(out[j]).asnumpy()
#             lab = int(label[j].asscalar())
#             best_probs = np.argmax(probs)
#             if lab == best_probs:
#                 total_correct += 1
#             total += 1
#     acc = total_correct / float(total)
#     return acc


if __name__ == '__main__':
    logging_config(args.log_dir, 'train', level=logging.INFO)
    vocab, train_dataset, val_dataset, test_dataset, transform = load_dataset(args.train_file, args.val_file, args.test_file)

    if args.embedding_source:
        # set embeddings as in PA1 or as appropriate for your approach
        if ':' in args.embedding_source:
            typ, src = tuple(args.embedding_source.split(':'))
            embedding = nlp.embedding.create(typ, source=src)
        else:
            print(args.embedding_source)
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

    ctx = mx.cpu()  ## or mx.gpu(N) if GPU device N is available
    train_classifier(vocab, transform, train_dataset, val_dataset, test_dataset, ctx)
