##for test
import sys
import math
import re

# import argparse
counter = {}
map_y = {"CCAT": 0, "ECAT": 0, "GCAT": 0, "MCAT": 0}


def tokenizeDoc(cur_doc):
    return re.findall('\\w+', cur_doc)


def log_Pr(counter, y, token):
    """
    Args:
        counter: the hashmap
        y: the specific y given
        tokens: tokens of the document
    return: the log probablity of the token given y
    """
    key_w = "Y=" + y + ",W=" + token
    key_y = "Y=" + y + ",W=*"
    return math.log((int(counter.get(key_w, 0)) + 1) / float(counter.get(key_y) + counter.get("W=*")))


def get_best_label(map_y, counter, tokens):
    """
    Args:
        map_y: the dom of y
        counter: the hashmap
        tokens: tokens of the document
    return: the most possibble label and its log probability
    """
    Y = ""
    P = float('-inf')
    for y in map_y:
        p = 0
        for token in tokens:
            p += log_Pr(counter, y, token)
        p += math.log((int(counter["Y=" + y]) + 1) / (float(counter.get["Y=" + y + ",W=*"]) + 4))
        if p > P:
            P = p
            Y = y
    return Y, str(P)


if __name__ == "__main__":
    N_test = 0
    N_correct = 0
    for line in sys.stdin:
        key, value = line.split("\t", 1)
        counter[key] = value
    with open(sys.argv[1], 'rb') as tests:
        for test in tests:
            N_test += 1
            ys, doc = test.split("\t", 1)
            ys = ys.split(",")
            tokens = tokenizeDoc(doc)
            label, log_prob = get_best_label(map_y, counter, tokens)
            if label in ys:
                N_correct += 1
            print ys + " " + label + " " + log_prob

    print "Percent correct: {0}/{1}={2}".format(N_correct, N_test, N_correct / float(N_test))