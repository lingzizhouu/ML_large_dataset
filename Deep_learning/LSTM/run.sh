vary_batch() {
	python lstm.py --max_len $1 --num_hid 50 --batch_size 64 --dataset smaller --epochs 25 --init_lr 0.5 --output_file output --train_loss_file train_loss
}
array=(10 15 20)
for item in ${array[*]}
do vary_batch $item
done
