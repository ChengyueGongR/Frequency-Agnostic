for %%i in (0.1 1.0 5.0 10.0 20.0) do (

	for %%j in (0.005 0.01 0.1) do (

		for %%k in (1 5 10 20) do (
	
				mkdir F:\users\dihe\fairseq_adv\ckpts\IWSLT14_DE_EN\lambda_%%i_lr_%%j_round_%%k_large
				copy F:\users\dihe\fairseq_adv\ckpts\IWSLT14_DE_EN\checkpoint_best_dec_single.pt F:\users\dihe\fairseq_adv\ckpts\IWSLT14_DE_EN\lambda_%%i_lr_%%j_round_%%k_large\checkpoint_last.pt
		
				python train.py data\iwslt14.tokenized.de-en --clip-norm 0.0 --min-lr 1e-09  --max-tokens 8192 --share-decoder-input-output-embed --arch transformer_iwslt_de_en --save-dir ckpts\IWSLT14_DE_EN\lambda_%%i_lr_%%j_round_%%k_large --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 --lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --lr 0.001 --warmup-init-lr 1e-07 --warmup-updates 4000 --max-update 20000 --adv_bias 3000 --adv_lambda %%i --adv_lr %%j --adv_updates %%k --no-progress-bar > lambda_%%i_lr_%%j_round_%%k_large.log
		)
	)
)

