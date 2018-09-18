## Acknowledgements

A large portion of this repo is borrowed from the following repos: https://github.com/salesforce/awd-lstm-lm

## Experiments

### WikiText-2 (WT2) with LSTM

+ `python main.py --epochs 750 --data data/wikitext-2 --dropouth 0.2 --seed 1882 --save ./trained_model/wiki2.pt --log-file ./log/wiki2_$IDX.log --alpha 0 --beta 1 --bptt 70`
+ `python finetune.py --epochs 750 --data data/wikitext-2 --save ./trained_model/finetune_wiki2.pt --log-file ./log/finetune_wiki2.log --dropouth 0.2 --seed 1882`
+ `python pointer.py --save ./trained_model/finetune_wiki2.pt --lambdasm 0.16 --theta 1.4 --window 4200 --bptt 2000 --data data/wikitext-2`

### Penn Treebank (PTB) with LSTM

+ `python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt --log-file ptb.log`
+ `python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 500 --save PTB.pt --log-file finetune_ptb.log`
+ `python pointer.py --data data/penn --save PTB.pt --lambdasm 0.09 --theta 0.75 --window 700 --bptt 5000`

## Further Work

Except adversarial regularization, we also test other methods which can match two distribution, e.g. MMD, empirical first n moment, neural mutmal information, empirical wasserstein distance. We notice most of them can work for language modeling and text classification. 

