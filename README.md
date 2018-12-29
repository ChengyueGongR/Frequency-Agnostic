# [Frequency Agnostic Word Representation](https://arxiv.org/pdf/1809.06858.pdf)
This is the code we used in our NIPS 2018 paper 
>Frequency-Agnostic Word Representation (Improving Word Embedding by Adversarial Training)

>Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-yan Liu

## Acknowledgements

A large portion of this repo is borrowed from the following repos:
https://github.com/salesforce/awd-lstm-lm, https://github.com/zihangdai/mos, https://github.com/pytorch/fairseq and https://github.com/tensorflow/tensor2tensor.

## Experiments
The hyper-parameters are set for `pytorch 0.3` version, and there may be some changes for `pytorch 0.4` version. (Also, the post-process should be changed for `pytorch 0.4`)
We have also seen exact reproduction numbers change when changing underlying GPU.
Therefore, the guide below produces results similar to the numbers reported. If you have some difficulties at reproducing the final results, just ask the first author for help (e-mail: cygong@pky.edu.cn)

## Word level WikiText-2 (WT2) with AWD-LSTM
Run the following commands:

+ `python main.py --epochs 1000 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --dropouti 0.55 --seed 1882`
+ `python finetune.py --epochs 1000 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --dropouti 0.55 --seed 1882`
+ `python pointer.py --save WT2.pt --lambdasm 0.16 --theta 1.4 --window 4200 --bptt 2000 --data data/wikitext-2`
