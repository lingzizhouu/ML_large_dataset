#!/usr/bin/env python
import re
import sys


def tokenizeDoc(cur_doc):
    return re.findall('\\w+', cur_doc)


def map_to_counter(tokens, label):
    for token in tokens:
        #word_set.add(token)
        #print '{0}\t{1}'.format(token, 1)
        #item = "Y=" + label + ",W=" + token
        # count the word occur under a specific label
        print 'Y=%s,W=%s\t%d' %(label, token, 1)
        # count the number of word occur under a specific label
        print 'Y=%s,W=*\t%d' %(label, 1)
    return


# count the unique word in the training set
#word_set = set()
for line in sys.stdin:
    line = line.strip()
# 3 items expected
    id, labels, doc = line.split("\t")

    for label in labels.split(','):
        # count the number of labels occured
        print 'Y=*\t%d' %(1)
        # count the number of specific label occured
        print 'Y=%s\t%d' %(label, 1)
        map_to_counter(tokenizeDoc(doc), label)
