# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion

import torch.nn as nn

@register_criterion('adv_embedding')
class AdvEmbeddingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        adv_h = model.adv_hidden(model.encoder.embed_tokens.weight)
        adv_criterion = nn.CrossEntropyLoss()
        adv_loss = adv_criterion(adv_h, self.adv_targets)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }

        return adv_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'sample_size': sample_size,
        }
        return agg_output
