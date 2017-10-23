#!/bin/bash
vary_mu () {
	for((i=1;i<=20;i++))
	do gshuf abstract.small.train;
done | python LR.py 10000 $1 0.1 10 40433 abstract.small.test
}
array=(0 1e-5 1e-4 1e-3 1e-2 1e-1 0.2 0.5 1)
for item in ${array[*]}
do vary_mu $item
done