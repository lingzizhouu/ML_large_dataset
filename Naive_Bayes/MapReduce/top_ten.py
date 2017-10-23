import sys
import collections

if __name__ == "__main__":
    # find the top ten words for each category
    dic = collections.defaultdict(list)
    for line in sys.stdin:
        line = line.strip()
        if "," not in line:
            continue
        class_name = line.split(",")[0][2:]
        word = line.split(",")[1].split("\t")[0][2:]
        count = line.split("\t")[-1]

        if word != "*":
            dic[class_name].append((word, count))

    for key, val in dic.items():
        val = sorted(val, lambda en: -1*en[1])
        end = max(len(val), 10)
        for tup in val[0: end]:
            print "%s\t%s\t%d" % (key, tup[0], tup[1])
            #print (key + "\t" + tup[0] + "\t" + str(tup[1]))



