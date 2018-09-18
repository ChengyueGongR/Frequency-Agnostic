# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

import torch.nn as nn

@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.adv_lambda = args.adv_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)

        #adv_h_enc = model.adv_hidden_enc(model.encoder.embed_tokens.weight)
        adv_criterion = nn.CrossEntropyLoss()
        #adv_loss_enc1 = adv_criterion(adv_h_enc[0:model.adv_bias_enc], model.adv_targets_enc[0:model.adv_bias_enc])
        #adv_loss_enc2 = adv_criterion(adv_h_enc[model.adv_bias_enc:], model.adv_targets_enc[model.adv_bias_enc:])
        #adv_loss_enc = (adv_loss_enc1 + adv_loss_enc2) / 2.0

        adv_h = model.adv_hidden(model.decoder.embed_tokens.weight)
        adv_loss1 = adv_criterion(adv_h[0:model.adv_bias], model.adv_targets[0:model.adv_bias])
        adv_loss2 = adv_criterion(adv_h[model.adv_bias:], model.adv_targets[model.adv_bias:])
        adv_loss = (adv_loss1 + adv_loss2) / 2.0

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss - self.adv_lambda * adv_loss  * sample_size
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
