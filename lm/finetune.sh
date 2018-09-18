IDX=1
#CUDA_VISIBLE_DEVICES=1 python finetune.py --epochs 750 --data data/wikitext-2 --dropouth 0.2 --seed 1882 --save ./trained_model/finetune_wiki2_$IDX.pt 
#--log-file ./log/finetune_wiki2_$IDX.log >>  ./tmp.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python finetune.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 800 --save trained_model/finetune_ptb_$IDX.pt --mmd_lambda 0.000 --kernel_alpha 2.0 --log-file ./log/finetune_ptb_$IDX.log  
#CUDA_VISIBLE_DEVICES=2 nohup python main.py --epochs 15 --nlayers 2 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 \
# --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 0 --bptt 140 --batch_size 20 --lr 1e-3 --data data/wikitext-103 --save WT103.pt --save ./trained_model/WK103_$IDX.pt --log-file ./log/wiki103_$IDX.log --gamma_h 0.5 --gamma_w 0 >>  ./tmp.log 2>&1 &


