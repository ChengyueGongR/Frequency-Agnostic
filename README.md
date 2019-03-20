# [Frequency Agnostic Word Representation](https://arxiv.org/pdf/1809.06858.pdf)
This is the code we used in our NIPS 2018 paper 
>Frequency-Agnostic Word Representation (Improving Word Embedding by Adversarial Training)

>Chengyue Gong, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-yan Liu

## Experiments
The hyper-parameters are set for `pytorch 0.4` version. The result of our paper is based on `pytorch 0.2`, so some results are better (awd-lstm) while some results are worse (awd-lstm-mos).

The performance will change when changing GPU.

Therefore, the guide below can produce results similar to the numbers reported, but maybe not exact. If you have some difficulties at reproducing the final results, feel free to ask the first author for help (e-mail: cygong@pku.edu.cn)

### Word level WikiText-2 (WT2) with AWD-LSTM
Run the following commands:

+ `python3 -u main.py --epochs 4000 --nonmono 5 --data data/wikitext-2 --dropouti 0.5 --dropouth 0.2 --seed 1882 --save ./trained_model/wiki2.pt --moment_split 8000 --moment_lambda 0.02`
or `python3 main.py --epochs 4000  --nonmono 5 --data data/wikitext-2 --save WT2.pt --dropouth 0.2 --dropouti 0.5 --seed 1882 --adv_split 8000 --adv_lambda 0.02`
+ `python3 pointer.py --save WT2.pt --lambdasm 0.16 --theta 1.4 --window 4200 --bptt 2000 --data data/wikitext-2`

### Word level Penn Treebank (PTB) with AWD-LSTM
You can download the pretrained model and the code here: https://drive.google.com/open?id=1x0GL8oYv21lwHkAkyWWgL7ViBRjxFAnc, the PPL after finetune is `57.7`/`55.8` (valid / test).

Run the following commands:

+ `python3 main.py --nonmono 5 --batch_size 20 --data data/penn --dropouth 0.25 --dropouti 0.3 --dropout 0.4 --alpha 2 --beta 1 --seed 141 --epoch 2000 --save ./trained_model/ptb.pt`
+ `python3 pointer.py --data data/penn --save PTB.pt --lambdasm 0.09 --theta 0.75 --window 700 --bptt 5000`

### Word level Penn Treebank (PTB) with AWD-LSTM-MoS
For the `pytroch 0.4.0`code, detailed information can be found in https://github.com/ChengyueGongR/Frequency-Agnostic/issues/2.
We can now achieve 56.00/53.82 after finetuning (it's 55.51/53.31 in our paper). 
You can download the pretrained model and the code here: https://drive.google.com/open?id=1znF6vrwNOXzWFS5KuIPVKGEEKoK5Fs57. The path for the final model is `./pretrained_ptb/finetune_model.pt`. 

## Acknowledgements

A large portion of this repo is borrowed from the following repos:
https://github.com/salesforce/awd-lstm-lm, https://github.com/zihangdai/mos, https://github.com/pytorch/fairseq and https://github.com/tensorflow/tensor2tensor.

Thanks @simtony, @takase and other gays for their useful advices.
