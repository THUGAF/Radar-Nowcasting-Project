python -u main_ef.py \
	--train \
	--test \
	--batch-size 50 \
	--data 140 \
	--epochs 50 \
	--freq 10 \
	--gpu 2 \
	> out.txt 2>&1 &
