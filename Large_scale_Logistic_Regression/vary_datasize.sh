#!/bin/bash
vary_datasize () {
	shuf abstract.$1.train | python LR.py 100000 0.5 0.1 10 2027540 abstract.$1.test
}
array=(tiny smaller small medium large full)
for item in ${array[*]}
do vary_datasize $item
done
