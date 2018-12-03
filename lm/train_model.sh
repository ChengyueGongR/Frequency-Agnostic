IDX=2
#CUDA_VISIBLE_DEVICES=3 nohup python3 main.py --epochs 750 --data data/wikitext-2 --dropouti 0.5 --nonmono 6 --dropouth 0.2 --seed 1882 --save ./trained_model/wiki2_$IDX.pt --log-file ./log/wiki2_$IDX.log  >>  ./tmp.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 main.py --nonmono 8 --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 1200 --save ./trained_model/ptb_$IDX.pt --log-file ./log/ptb_$IDX.log >>  ./tmp.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python main.py --epochs 15 --nlayers 3 --emsize 128 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 \
# --dropouti 0.1 --dropout 0.1 --wdrop 0.5 --wdecay 0 --bptt 120 --batch_size 40 --lr 1e-3 --data data/wikitext-103 --save ./trained_model/WK103_$IDX.pt --log-file ./log/wiki103_$IDX.log --gamma_h 0.5 --gamma_w 0 >>  ./tmp.log 2>&1 &


