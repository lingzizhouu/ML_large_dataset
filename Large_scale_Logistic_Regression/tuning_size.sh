#!/bin/bash
vary_size () {
	for((i=1;i<=20;i++))
	do gshuf abstract.small.train;
done | python LR.py $1 0.5 0.1 10 40433 abstract.small.test
}
array=(10 100 1000 10000 100000)
for item in ${array[*]}
do vary_size $item
done