edges_file := ../data/freebase_1.graph
seeds_file := ../data/seed1_2.txt
eval_file := ../data/eval1_2.txt
gold_file := ../data/gold1_2.txt
iterations := 2
number_of_excutors := 4


run:
	spark-submit lp.py --edges_file $(edges_file) --seeds_file $(seeds_file) --eval_file $(eval_file) --iterations $(iterations) --number_of_excutors $(number_of_excutors) > result.txt
	
eval:
	python mrr.py --n_labels 23 --output_file result.txt --gold_file $(gold_file)

handin:
	tar cvf hw6.tar lp.py
