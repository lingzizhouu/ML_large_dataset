import argparse
import sys
from pyspark import SparkContext, SparkConf
from operator import add
import logging
logger = logging.getLogger('data-producer')
logger.setLevel(logging.DEBUG)

def create_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--iterations', type=int, default=2,
                      help='Number of iterations of label propagation')
  parser.add_argument('--edges_file', default=None,
                      help='Input file of edges')
  parser.add_argument('--seeds_file', default=None,
                      help='File that contains labels for seed nodes')
  parser.add_argument('--eval_file', default=None,
                      help='File that contains labels of nodes to be evaluated')
  parser.add_argument('--number_of_excutors', type=int, default=8,
                      help='Number of iterations of label propagation')
  return parser


def parse_edge(edge):
    """Parses a graph  pair string into urls pair."""
    entity_node, property_node, weight = edge[0], edge[1], edge[2]
    yield entity_node, (property_node, float(weight))
    yield property_node, (entity_node, float(weight))


def seed_map(seed):
    node, label = seed.split()
    return node, (label, 1)


def reset_seed(x):
    """
    :param x: joined unique node with seed
    e.g. (u'prime', ((), (u'superheros', 1))), (u'to_ride_pegasus', (((u'superheros', 0.2), ()), None))
    :return:
    e.g. (u'graham_chapman', (u'composers', 1)) for seed node, unchanged for non-seed node
    """
    if x[1][1]:
        return x[0], x[1][1]
    else:
        return x[0], x[1][0]


def exist_tuple_of_tuple(t_list):
    return any(isinstance(e, tuple) for e in t_list)


def transition_normalize(x):
    node, edge_list = x
    weight_sum = 0
    row = list(edge_list)
    normalized_row = [None]*len(row)
    if exist_tuple_of_tuple(edge_list):
        for edge in row:
            weight_sum += edge[1]
        for i in range(len(row)):
            normalized_row[i] = row[i][0], row[i][1] / weight_sum
    else:
        normalized_row = edge_list[0], 1

    return node, tuple(normalized_row)


def update_rank(target_node):
    # Todo update the rank of the given node
    """
    :param node: (u'keyport', (<pyspark.resultiterable.ResultIterable object at 0x13da0050>, ()))
    old_rank: ranks for different label
    linked_edges: linked edges of the node
    :return:
    """
    node, (linked_edges, old_ranks) = target_node
    # with more than one linked_edges
    if exist_tuple_of_tuple(linked_edges):
        for linked_edge in linked_edges:
            linked_node, weight = linked_edge
            if not old_ranks:
                return
            if exist_tuple_of_tuple(old_ranks):
                for old_rank in old_ranks:
                    label, label_rank = old_rank
                    yield linked_node, (label, weight * label_rank)
            else:
                label, label_rank = old_ranks
                yield linked_node, (label, weight * label_rank)
    # with only one linked_edge
    else:
        linked_node, weight = linked_edges
        if not old_ranks:
            return
        if exist_tuple_of_tuple(old_ranks):
            for old_rank in old_ranks:
                label, label_rank = old_rank
                yield linked_node, (label, weight * label_rank)
        else:
            label, label_rank = old_ranks
            yield linked_node, (label, weight * label_rank)


def aggregate_and_normalize(updation):
    node, label_updations = updation
    updation_map = {}
    updation_sum = 0
    if exist_tuple_of_tuple(label_updations):
        for label_updation in label_updations:
            label, count = label_updation
            updation_sum += count
            if label in updation_map:
                updation_map[label] += count
            else:
                updation_map[label] = count
    else:
        label, count = label_updations
        updation_sum += count
        if label in updation_map:
            updation_map[label] += count
        else:
            updation_map[label] = count
    label_ranks = [None]*len(updation_map)
    for i, (k, v) in enumerate(updation_map.iteritems()):
        label_ranks[i] = (k, v / float(updation_sum))
    return node, tuple(label_ranks)


def get_complete_rank(r, unique_labels):
    node, (_, ranks) = r
    rank_list = []
    if ranks and exist_tuple_of_tuple(ranks):
        rank_list = list(ranks)
        for label in unique_labels:
            if label not in rank_list:
                rank_list.append((label, 0))
    else:
        if ranks:
            rank_list = [ranks]
        else:
            rank_list = []
        for label in unique_labels:
            if label not in rank_list:
                rank_list.append((label, 0))

    return node, sorted(rank_list, key=lambda x: -x[1])


class LabelPropagation:
    def __init__(self, graph_file, seed_file, eval_file, iterations, number_of_excutors):
        conf = SparkConf().setAppName("LabelPropagation")
        conf = conf.setMaster('local[%d]'% number_of_excutors)\
                 .set('spark.executor.memory', '8G')\
                 .set('spark.driver.memory', '8G')\
                 .set('spark.driver.maxResultSize', '8G')
        self.spark = SparkContext(conf=conf)
        self.graph_file = graph_file
        self.seed_file = seed_file
        self.eval_file = eval_file
        self.n_iterations = iterations
        self.n_partitions = number_of_excutors * 2

    def run(self):
        lines = self.spark.textFile(self.graph_file, self.n_partitions)
        directed_edges = lines.map(lambda line: line.split('\t'))
        undirected_edges = directed_edges.flatMap(lambda edge: parse_edge(edge))
        unique_nodes = undirected_edges.map(lambda edge: (edge[0], ())).distinct()
        # store the node and the corresponding out-edges
        node_outedges = undirected_edges.groupByKey().map(lambda x: transition_normalize(x))
        # print node_outedges.map(lambda x: (x[0], list(x[1]))).collect()
        # Todo initialize the rank score of the node for each label, all 0 except the seeds
        lines = self.spark.textFile(self.seed_file, self.n_partitions)
        unique_labels = lines.map(lambda line: line.split('\t')[1]).distinct().collect()
        seed_rank = lines.map(lambda line: seed_map(line)).cache()
        #print unique_nodes.collect()

        rank = unique_nodes.leftOuterJoin(seed_rank).map(lambda x: reset_seed(x))


        # Todo join the rank score with the edges to generate the new rank score

        for t in range(self.n_iterations):
            # update the rank
            rank_updation = node_outedges.join(rank).flatMap(lambda record: update_rank(record))
            # group and row normalization
            unclamped_rank = rank_updation.groupByKey().map(lambda updation: aggregate_and_normalize(updation))
            rank = unclamped_rank.leftOuterJoin(seed_rank).map(lambda x: reset_seed(x))

        # [TODO]
        #print rank.collect()
        lines = self.spark.textFile(self.eval_file, self.n_partitions)
        lines = lines.map(lambda line: (line, 1))
        # print lines.collect()
        predictions = lines.leftOuterJoin(rank).map(lambda x: get_complete_rank(x, unique_labels))
        ans = predictions.collect()
        for a in ans:
            node, label_ranks = a
            write_line = node
            for label, score in label_ranks:
                write_line += '\t' + label.strip()
            print write_line.encode('utf-8')
        return

    def eval(self):
        pass

if __name__ == "__main__":
    args = create_parser().parse_args()
    lp = LabelPropagation(args.edges_file, args.seeds_file, args.eval_file, args.iterations, args.number_of_excutors)
    lp.run()
    lp.eval()
    lp.spark.stop()
