#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import itertools
import os
import math
import torch
import numpy as np

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def low_rank_approx(A, r):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    u, s, v = np.linalg.svd(A, full_matrices=False)

    Ar = np.zeros((len(u), r))

    for i in range(r):
        Ar[:, i] = s[i] * u[:, i]

    return Ar, s

def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Build trainer
    if args.fp16:
        trainer = FP16Trainer(args, task, model, criterion)
    else:
        if torch.cuda.get_device_capability(0)[0] >= 7:
            print('| NOTICE: your device may support faster training with --fp16')
        trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    max_positions = trainer.get_model().max_positions()
    epoch_itr = data.EpochBatchIterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # Load the latest checkpoint if one is available
    load_checkpoint(args, trainer, epoch_itr)
    embedding = model.decoder.embed_tokens.weight.data.cpu().numpy()
    print (embedding.shape)

    Ar,s = low_rank_approx(embedding,2)
    print(Ar.shape)

    np.savetxt('svd', Ar, delimiter=' ')

def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
