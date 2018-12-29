# [Frequency Agnostic Word Representation](https://arxiv.org/pdf/1809.06858.pdf)
This is the code we used in our NIPS 2018 paper 
>Frequency-Agnostic Word Representation (Improving Word Embedding by Adversarial Training)

>Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-yan Liu

## Experiments
The hyper-parameters are set for `pytorch 0.3` version, and there may be some changes for `pytorch 0.4` version. 

Also, the performance will change when changing GPU.

Therefore, the guide below can produce results similar to the numbers reported, but maybe not exact. If you have some difficulties at reproducing the final results, feel free to ask the first author for help (e-mail: cygong@pku.edu.cn)

### Word level WikiText-2 (WT2) with AWD-LSTM
Run the following commands:

+ `python main.py --epochs 1000 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --dropouti 0.55 --nonmono 15 --seed 1882`
or `python main.py --epochs 1000 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --dropouti 0.55 --nonmono 15 --seed 1882 --moment --adv --adv_split 8000 --adv_lambda 0.02`
+ `python finetune.py --epochs 1000 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --dropouti 0.55 --seed 1882`
+ `python pointer.py --save WT2.pt --lambdasm 0.16 --theta 1.4 --window 4200 --bptt 2000 --data data/wikitext-2`

### Word level Penn Treebank (PTB) with AWD-LSTM
Run the following commands:

+ `python main.py --batch_size 20 --data data/penn --dropouti 0.3 --dropouth 0.25 --seed 141 --nonmono 15 --epoch 800 --save PTB.pt --moment_split 1000 --moment_lambda 0.1`
+ `python finetune.py --batch_size 20 --data data/penn --dropouti 0.3 --dropouth 0.25 --seed 141 --epoch 800 --save PTB.pt`
+ `python pointer.py --data data/penn --save PTB.pt --lambdasm 0.09 --theta 0.75 --window 700 --bptt 5000`

## Acknowledgements

A large portion of this repo is borrowed from the following repos:
https://github.com/salesforce/awd-lstm-lm, https://github.com/zihangdai/mos, https://github.com/pytorch/fairseq and https://github.com/tensorflow/tensor2tensor.
