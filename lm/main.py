import argparse
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils import batchify, get_batch, repackage_hidden, message, set_log_file

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--gamma_h', type=float, default=0.1,
                    help='gamma L2 regularization of the Normalized RNN activation (lambda = 0 means no regularization')
parser.add_argument('--gamma_w', type=float, default=0.1,
                    help='gamma L2 regularization of the Normalized RNN activation (lambda = 0 means no regularization')


parser.add_argument('--log-file', type=str,  default='',
                    help='path to save the log')
parser.add_argument('--mmd_kernel_alpha', type=float,  default=0.5,
                    help='mmd kernel')
parser.add_argument('--mmd_lambda', type=float,  default=0.2,
                    help='mmd kernel')
parser.add_argument('--moment', action='store_false',
                    help='using moment regularization')
parser.add_argument('--moment_split', type=int, default=8000,
                    help='threshold for rare and popular words')
parser.add_argument('--moment_lambda', type=int, default=0.1,
                    help='lambda')
parser.add_argument('--adv', action='store_true',
                    help='using adversarial regularization')
parser.add_argument('--adv_bias', type=int, default=8000,
                    help='threshold for rare and popular words')
parser.add_argument('--adv_lambda', type=int, default=0.1,
                    help='lambda')
parser.add_argument('--adv_lr', type=float,  default=0.02,
                    help='adv learning rate')
parser.add_argument('--adv_wdecay', type=float,  default=1.2e-6,
                    help='adv weight decay')

args = parser.parse_args()
#args.tied = False
set_log_file(args.log_file)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        message("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
    model.encoder_sigma.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
message('Args: ' + repr(args))
message('Model total parameters: ' + repr(total_params))

criterion = nn.CrossEntropyLoss()
#if args.adv:
#    rate = (ntokens - args.adv_bias) * 1.0 / ntokens
#    adv_criterion = nn.CrossEntropyLoss(weight=torch.Tensor([rate, 1 - rate]).cuda())
#    adv_hidden = nn.Linear(args.emsize, 2).cuda()
#    adv_targets = torch.LongTensor(np.array([0] * args.adv_bias + [1] * (ntokens - args.adv_bias))).cuda()
#    adv_targets = Variable(adv_targets)
#    adv_hidden.weight.data.uniform_(-0.1, 0.1)
###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def train(flag):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    #total_norm = 0
    #total_w_norm = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.

        if args.moment:
           hidden = repackage_hidden(hidden)
           optimizer.zero_grad()

           output, hidden, rnn_hs, dropped_rnn_hs, w = model(data, hidden, return_h=True)
           raw_loss = criterion(output.view(-1, ntokens), targets)

           bias = args.moment_split
           common = model.encoder.weight[:bias].detach()
           rare = model.encoder.weight[bias:]
           mean0 = torch.mean(common, 0)
           mean1 = torch.mean(rare, 0)
           var0 = torch.var(common, 0)
           var1 = torch.var(rare, 0)
           kewness0 = torch.mean(torch.pow(common - mean0, 3), 0) / torch.pow(var0, 1.5)
           kewness1 = torch.mean(torch.pow(rare - mean1, 3), 0) / torch.pow(var1, 1.5)
           kurtosis0 = torch.mean(torch.pow(common - mean0, 4), 0) / torch.pow(var0, 2)
           kurtosis1 = torch.mean(torch.pow(rare - mean1, 4), 0) / torch.pow(var1, 2)
           reg_loss = torch.sqrt(torch.sum(torch.pow(mean0 - mean1, 2))) + torch.sqrt(torch.sum(torch.pow(var0 - var1, 2))) \
                      + torch.sqrt(torch.sum(torch.pow(kewness0 - kewness1, 2))) + torch.sqrt(torch.sum(torch.pow(kurtosis0 - kurtosis1, 2)))
           loss = raw_loss + args.moment_lambda * reg_loss
        elif args.adv:
           # calculate the adv_classifier
           optimizer.zero_grad()
           adv_optimizer.zero_grad()
           adv_h = adv_hidden(model.encoder.weight)
           adv_loss = adv_criterion(adv_h, adv_targets)
           adv_loss.backward()
           adv_optimizer.step()

           hidden = repackage_hidden(hidden)
           adv_optimizer.zero_grad()
           optimizer.zero_grad()
           output, hidden, rnn_hs, dropped_rnn_hs, w = model(data, hidden, return_h=True)
           raw_loss = criterion(output.view(-1, ntokens), targets)

           adv_h = adv_hidden(model.encoder.weight)
           adv_loss = adv_criterion(adv_h, adv_targets)
           loss = raw_loss - args.adv_lambda * adv_loss
        else:
           loss = raw_loss
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs, w = model(data, hidden, return_h=True, is_switch=switch)
        raw_loss = criterion(output.view(-1, ntokens), targets)
        loss = raw_loss
        # Activiation Regularization
        loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:]) 
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        #if switch:
        #    aux_optimizer.step()
        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            #cur_norm = total_norm[0] / args.log_interval
            #cur_w_norm = total_w_norm[0] / args.log_interval
            elapsed = time.time() - start_time
            message('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} '.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            #total_norm = 0
            #total_w_norm = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
switch=False

# At any point you can hit Ctrl + C to break out of training early.
try:
    #if args.adv:
    #    adv_optimizer = torch.optim.SGD(adv_hidden.parameters(), lr=args.adv_lr, weight_decay=args.adv_wdecay)
    optimizer = torch.optim.SGD(list(model.parameters()), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        #print(f'Save to {args.save}')
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()
            val_loss2 = evaluate(val_data)
            message('-' * 89)
            message('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            message('-' * 89)
            test_loss = evaluate(test_data, test_batch_size)
            # message('=' * 89)
            message('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
            message('=' * 89)

            if val_loss2 < stored_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                message('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            message('-' * 89)
            message('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            message('-' * 89)
            test_loss = evaluate(test_data, test_batch_size)
            # message('=' * 89)
            message('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, math.exp(test_loss)))
            message('=' * 89)

            if val_loss < stored_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                message('Saving Normal!')
                stored_loss = val_loss

            if epoch >= 120:
            # if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                message('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #aux_optimizer = torch.optim.SGD(list(model.encoder[0].parameters())+list(model.encoder_sigma[0].parameters()), lr=args.lr,  weight_decay=args.wdecay)#model.encoder_sigma[0].parameters()
                switch=True#optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    message('-' * 89)
    message('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
message('=' * 89)
message('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
message('=' * 89)
