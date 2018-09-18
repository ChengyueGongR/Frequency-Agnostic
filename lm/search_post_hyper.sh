alpha_list="0.16"
beta_list="4200 4500 5000"
#alpha_list="1.3 1.4"
for alpha in ${alpha_list}
do
#  CUDA_VISIBLE_DEVICES=1 python pointer.py --save trained_model/wiki2_11.pt --lambdasm ${alpha} --theta 0.75 --window 4200 --bptt 2000 --data data/wikitext-2 
  CUDA_VISIBLE_DEVICES=1 python pointer.py --data data/penn --save trained_model/finetune_ptb_15.pt --lambdasm 0.09 --theta 1.4 --window 700 --bptt 5000
done


