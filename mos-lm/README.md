## Acknowledgements

A large portion of this repo is borrowed from the following repos:
https://github.com/benkrause/dynamic-evaluation and https://github.com/zihangdai/mos.

## Bug
We have fixed the bug. Now, the number on PTB dataset is about 57.7/55.4(valid/test) w/o finetune. (in the paper, 57.5/55.2)

Also, if you notice finetune stage doesn't work, please try to use first SGD then repeat ASGD. 
