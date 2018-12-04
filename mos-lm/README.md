## Acknowledgements

A large portion of this repo is borrowed from the following repos:
https://github.com/benkrause/dynamic-evaluation and https://github.com/zihangdai/mos.

## Bug
We notice the result of AWD-LSTM-MoS on PTB dataset will drop, and we are trying to fix this bug.

For WT2 dataset, use --moment setting, you can get similar results as reported in the paper.

Also, if you notice finetune stage doesn't work, please try to use first SGD then ASGD as in training period.
