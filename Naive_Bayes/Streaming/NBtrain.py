#!usr/bin/env python
import re
import sys

def tokenizeDoc(cur_doc):
     return re.findall('\\w+', cur_doc)

def map_to_counter(word_set, tokens, label, counter, count_word):
    for token in tokens:
        word_set.add(token)
        item = "Y=" + label + ",W=" + token
        counter[item] = counter.get(item, 0) + 1
        count_word[label] = count_word.get(label, 0) + 1
    return

map_y = {"CCAT" : 0, "ECAT" : 0, "GCAT" : 0, "MCAT" : 0}
#count the number of CAT labels occured
count_y = 0;
#count the number of specific word of given label occured
counter = {}
#count the total number of the words occur for a label
count_word = {}
# count the unique word in the training set
word_set = set()
if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        labels, doc =  line.split("\t", 1)
        
        for label in labels.split(','):
            if label in map_y:
                count_y += 1
                map_y[label] += 1
                map_to_counter(word_set, tokenizeDoc(doc), label, counter, count_word)

#std_output
for key, value in map_y.iteritems():
    print "Y=" + key + "\t" + str(value)

for key, value in counter.iteritems():
    print key + "\t" + str(value)
    
for key, value in count_word.iteritems():
    print "Y=" + key + ",W=*" + "\t" + str(value)

print "Y=*" + "\t" + str(count_y)
print "W=*" + "\t" + str(len(word_set))
