src/lp.py                # label propagation
---/mrr.py               # The script which evaluates MRR of output file result.txt
---/Makefile             # makefile which can run spark code and generate hw6.tar





                         # Graph
data/freebase_1.graph    # A graph which contains 32K nodes and 957K edges
----/freebase_2.graph    # A graph which contains 301K nodes and 2.3M edges


                         # Seeds: The first column represents node
                         #        The second column represents label
----/seed1_2.txt         # Two seeds per class for freebase1
----/seed1_10.txt        # Ten seeds per class for freebase1
----/seed2_2.txt         # Two seeds per class for freebase2
----/seed2_10.txt        # Ten seeds per class for freebase2


                         # Evaluation: Each line contains a node to be evaluated
----/eval1_2.txt         # Evaluation file for seed1_2.txt
----/eval1_10.txt        # Evaluation file for seed1_10.txt
----/eval2_2.txt         # Evaluation file for seed2_2.txt
----/eval2_10.txt        # Evaluation file for seed2_10.txt


                         # Ground truths: The first column represents node
                                          The second column represents gold label
----/gold1_2.txt         # Ground truth file for seed1_2.txt
----/gold1_10.txt        # This file is not provided, since it will be evaluated on autolab
----/gold2_2.txt         # Ground truth file for seed2_2.txt
----/gold2_10.txt        # Ground truth file for seed2_10.txt


