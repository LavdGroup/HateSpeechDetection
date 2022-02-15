"""
run_bert.py : train, test, eval bert model.

Usage:
    run_bert.py train MODEL [options]
    run_bert.py test MODEL [options]

    MODEL is in [default, nonlinear, lstm, cnn]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train=<file>                          train file [default: train.csv]
    --dev=<file>                            dev file [default: val.csv]
    --test=<file>                           test file [default: test.csv]
    --seed=<int>                            seed [default: 42]
    --batch-size=<int>                      batch size [default: 32]
    --out-channel=<int>                     out channel for cnn [default: 16]
    --clip-grad=<float>                     gradient clipping [default: 1.0]
    --max-epoch=<int>                       max epoch [default: 3]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 3]
    --max-num-trial=<int>                   terminate training after how many trials [default: 3]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr-bert=<float>                       BERT learning rate [default: 0.00002]
    --lr=<float>                            learning rate [default: 0.001]
    --valid-niter=<int>                     perform validation after how many iterations [default: 500]
    --dropout=<float>                       dropout probability [default: 0.1]
    --verbose                               whether to output the test results
"""
from docopt import docopt
from pytorch_pretrained_bert import BertAdam
from bert_model import DefaultModel, NonlinearModel, CustomBertLSTMModel, CustomBertConvModel
import logging
import pickle
import numpy as np
import torch
import pandas as pd
import time
import sys
from utils import batch_iter
from sklearn.metrics import f1_score, precision_score, recall_score


def validation(model, df_val, loss_func, device):
    """ validation of model during training.
    @param model (nn.Module): the model being trained
    @param df_val (dataframe): validation dataset
    @param loss_func(nn.Module): loss function
    @param device (torch.device)

    @return avg loss value across validation dataset
    """
    was_training = model.training
    model.eval()  # evaluation mode

    ProcessedText_BERT = list(df_val.ProcessedText_BERT)
    Mapped_label = list(df_val.Mapped_label)

    val_batch_size = 32

    n_batch = int(np.ceil(df_val.shape[0]/val_batch_size))

    total_loss = 0.0

    with torch.no_grad():
        for i in range(n_batch):
            sents = ProcessedText_BERT[i*val_batch_size: (i+1)*val_batch_size]
            targets = torch.tensor(Mapped_label[i*val_batch_size: (i+1)*val_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)
            pre_softmax = model(sents).double()
            batch_loss = loss_func(pre_softmax, targets)
            total_loss += batch_loss.item()*batch_size

    if was_training:
        model.train()  # training mode

    return total_loss/df_val.shape[0]


def train(args):
    label_name = ['offensive', 'neither', 'hate']
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    prefix = args['MODEL']
    print('prefix: ', prefix)

    df_train = pd.read_csv(args['--train'], index_col=0)
    df_val = pd.read_csv(args['--dev'], index_col=0)
    train_label = dict(df_train.Mapped_label.value_counts())
    label_max = float(max(train_label.values()))
    train_label_weight = torch.tensor([label_max/train_label[i] for i in range(len(train_label))], device=device)  # give more weight to underrepresented classes

    if args['MODEL'] == 'default':
        model = DefaultModel(device, len(label_name))
        optimizer = BertAdam([
                {'params': model.bert.bert.parameters()},
                {'params': model.bert.classifier.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    elif args['MODEL'] == 'nonlinear':
        model = NonlinearModel(device, len(label_name), float(args['--dropout']))
        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.linear1.parameters(), 'lr': float(args['--lr'])},
                {'params': model.linear2.parameters(), 'lr': float(args['--lr'])},
                {'params': model.linear3.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    elif args['MODEL'] == 'lstm':
        model = CustomBertLSTMModel(device, float(args['--dropout']), len(label_name))
        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.lstm.parameters(), 'lr': float(args['--lr'])},
                {'params': model.hidden_to_softmax.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    elif args['MODEL'] == 'cnn':
        model = CustomBertConvModel(device, float(args['--dropout']), len(label_name),
                                    out_channel=int(args['--out-channel']))
        optimizer = BertAdam([
                {'params': model.bert.parameters()},
                {'params': model.conv.parameters(), 'lr': float(args['--lr'])},
                {'params': model.hidden_to_softmax.parameters(), 'lr': float(args['--lr'])}
            ], lr=float(args['--lr-bert']), max_grad_norm=float(args['--clip-grad']))
    else:
        print("invalid model")
        exit(0)

    model = model.to(device)
    print('-' * 80)

    model.train()

    cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight)
    torch.save(cn_loss, 'loss_func')  # for later testing

    train_batch_size = int(args['--batch-size'])
    valid_niter = int(args['--valid-niter'])
    model_save_path = prefix+'_model.bin'

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = 0
    cum_examples = report_examples = epoch = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Training: ')
    
    while True:
        epoch += 1

        for sents, targets in batch_iter(df_train, batch_size=train_batch_size, shuffle=True):  # for each epoch
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(sents)
            pre_softmax = model(sents).double() 

            loss = cn_loss(pre_softmax, torch.tensor(targets, dtype=torch.long, device=device))

            loss.backward()
            optimizer.step()

            batch_losses_val = loss.item() * batch_size
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_examples += batch_size
            cum_examples += batch_size



            # perform validation
            if train_iter % valid_niter == 0:

                cum_loss = cum_examples = 0.
                print('Validation')

                validation_loss = validation(model, df_val, cn_loss, device)

                print('validation: iter %d, loss %f' % (train_iter, validation_loss), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or validation_loss < min(hist_valid_scores)
                hist_valid_scores.append(validation_loss)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)

                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        print('load previously best model and decay learning rate to %f%%' %
                              (float(args['--lr-decay'])*100))

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers')
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= float(args['--lr-decay'])

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!')
                    print(f"Training time: {time.time() - train_time}")
                    exit(0)


def test(args):
    label_name = ['offensive', 'neither', 'hate'] 

    prefix = args['MODEL']

    bert_size = 768

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    print('load best model...')

    if args['MODEL'] == 'default':
        model = DefaultModel.load(prefix + '_model.bin', device)
    elif args['MODEL'] == 'nonlinear':
        model = NonlinearModel.load(prefix + '_model.bin', device)
    elif args['MODEL'] == 'lstm':
        model = CustomBertLSTMModel.load(prefix+'_model.bin', device)
    elif args['MODEL'] == 'cnn':
        model = CustomBertConvModel.load(prefix+'_model.bin', device)

    model.to(device)

    model.eval()

    df_test = pd.read_csv(args['--test'], index_col=0)

    df_test = df_test

    test_batch_size = 32

    n_batch = int(np.ceil(df_test.shape[0]/test_batch_size))

    cn_loss = torch.load('loss_func', map_location=lambda storage, loc: storage).to(device)

    ProcessedText_BERT = list(df_test.ProcessedText_BERT)
    Mapped_label = list(df_test.Mapped_label)

    test_loss = 0.
    prediction = []
    prob = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i in range(n_batch):
            sents = ProcessedText_BERT[i*test_batch_size: (i+1)*test_batch_size]
            targets = torch.tensor(Mapped_label[i * test_batch_size: (i + 1) * test_batch_size],
                                   dtype=torch.long, device=device)
            batch_size = len(sents)

            pre_softmax = model(sents).double()
            batch_loss = cn_loss(pre_softmax, targets)
            test_loss += batch_loss.item()*batch_size
            prob_batch = softmax(pre_softmax)
            prob.append(prob_batch)

            prediction.extend([t.item() for t in list(torch.argmax(prob_batch, dim=1))])

    prob = torch.cat(tuple(prob), dim=0)
    loss = test_loss/df_test.shape[0]
    pickle.dump([label_name[i] for i in prediction], open(prefix+'_test_prediction', 'wb'))
    pickle.dump(prob.data.cpu().numpy(), open(prefix + '_test_prediction_prob', 'wb'))
    
    precisions = {}
    recalls = {}
    f1s = {}
    
    weighted_f1 = f1_score(df_test.Mapped_label.values, prediction, average="weighted")
    
    for i in range(len(label_name)):
        prediction_ = [1 if pred == i else 0 for pred in prediction]
        true_ = [1 if label == i else 0 for label in df_test.Mapped_label.values]
        f1s.update({label_name[i]: f1_score(true_, prediction_)})
        precisions.update({label_name[i]: precision_score(true_, prediction_)})
        recalls.update({label_name[i]: recall_score(true_, prediction_)})

    metrics_dict = {'loss': loss, 'precision': precisions, 'recall': recalls, 'f1': f1s}

    pickle.dump(metrics_dict, open(prefix+'_evaluation_metrics', 'wb'))


    if args['--verbose']:
        print('loss: %.2f' % loss)
        print(f"weighted F1 {weighted_f1}")
        print('-' * 80)
        for i in range(len(label_name)):
            print('precision score for %s: %.2f' % (label_name[i], precisions[label_name[i]]))
            print('recall score for %s: %.2f' % (label_name[i], recalls[label_name[i]]))
            print('f1 score for %s: %.2f' % (label_name[i], f1s[label_name[i]]))
            print('-' * 80)


if __name__ == '__main__':
    args = docopt(__doc__)
    print(f"running run_bert.py with the following argumnets:\n{args}\n")

    logging.basicConfig(level=logging.INFO)

    if args['train']:
        train(args)

    elif args['test']:
        test(args)

    else:
        raise RuntimeError('invalid run mode')
