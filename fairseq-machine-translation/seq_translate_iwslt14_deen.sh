PROBLEM=IWSLT14_DE_EN
HYPERPARAMETER=lambda_5.0_lr_0.01_large
REMOTE_DATA_PATH=/var/storage/shared/msrmt/dihe/fairseq_adv/data
CODE_PATH=/var/storage/shared/msrmt/dihe/fairseq_adv
REMOTE_MODEL_PATH=/var/storage/shared/msrmt/dihe/fairseq_adv/ckpts

nvidia-smi

python -c "import torch; print(torch.__version__)"


#python ${CODE_PATH}/seq_translate.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en --ckpt-dir ${REMOTE_MODEL_PATH}/$model/$PROBLEM/${ARCH}_${SETTING} --initial-model ${INITIAL_MODEL_NAME} --batch-size 128 --beam 5 --lenpen ${LPEN} --quiet --remove-bpe --no-progress-bar | tee ${CODE_PATH}/log/trans/bleus/${model}_${PROBLEM}_${ARCH}_${SETTING}_${INITIAL_MODEL_NAME}_${LPEN}.txt


python ${CODE_PATH}/generate.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en --path ${REMOTE_MODEL_PATH}/$PROBLEM/$HYPERPARAMETER/checkpoint19.pt --batch-size 128 --beam 5 --lenpen 1.2 --quiet --remove-bpe --no-progress-bar 
python ${CODE_PATH}/generate.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en --path ${REMOTE_MODEL_PATH}/$PROBLEM/$HYPERPARAMETER/checkpoint20.pt --batch-size 128 --beam 5 --lenpen 1.2 --quiet --remove-bpe --no-progress-bar 
python ${CODE_PATH}/generate.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en --path ${REMOTE_MODEL_PATH}/$PROBLEM/$HYPERPARAMETER/checkpoint21.pt --batch-size 128 --beam 5 --lenpen 1.2 --quiet --remove-bpe --no-progress-bar 
python ${CODE_PATH}/generate.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en --path ${REMOTE_MODEL_PATH}/$PROBLEM/$HYPERPARAMETER/checkpoint22.pt --batch-size 128 --beam 5 --lenpen 1.2 --quiet --remove-bpe --no-progress-bar 
python ${CODE_PATH}/generate.py ${REMOTE_DATA_PATH}/iwslt14.tokenized.de-en --path ${REMOTE_MODEL_PATH}/$PROBLEM/$HYPERPARAMETER/checkpoint23.pt --batch-size 128 --beam 5 --lenpen 1.2 --quiet --remove-bpe --no-progress-bar 
