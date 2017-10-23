from guineapig import *
import re
import math
# supporting routines can go here
stop_word = set([u'all', u'just', u'being', u'over', u'both', u'through',\
                 u'yourselves', u'its', u'before', u'o', u'hadn', u'herself', \
                 u'll', u'had', u'should', u'to', u'only', u'won', u'under', u'ours', \
                 u'has', u'do', u'them', u'his', u'very', u'they', u'not', u'during', u'now', \
                 u'him', u'nor', u'd', u'did', u'didn', u'this', u'she', u'each', u'further', \
                 u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our', \
                 u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', \
                 u'between', u'mustn', u't', u'be', u'we', u'who', u'were', u'here', u'shouldn'\
                    , u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'isn', \
                 u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your', u'from', \
                 u'her', u'their', u'aren', u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', \
                 u'weren', u'was', u'until', u'more', u'himself', u'that', u'but', u'don', u'with', u'than', \
                 u'those', u'he', u'me', u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', \
                 u'my', u'and', u've', u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', \
                 u'any', u'if', u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', \
                 u'haven', u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having', u'once'])
def isanumber(a):

    try:
        float(repr(a))
        bool_a = True
    except:
        bool_a = False

    return bool_a


def tokenizeDoc(cur_doc):
    ans = []
    for w in re.findall('\\w+', cur_doc):
        if (w not in stop_word and not isanumber(w)):
            ans.append(w)
    return ans


def tokenize(doc):
    for tok in tokenizeDoc(doc): yield tok


def labelDocGen(line):
    id, labels, doc = line.strip().split('\t')
    for label in labels.split(','):
        yield (label, doc)


def WordIdGen(line):
    id, labels, doc = line.strip().split('\t')
    for word in tokenizeDoc(doc):
        yield (word, id)


def labelTokenGen((label, doc)):
    for tok in tokenizeDoc(doc): yield (label, tok)


def bestLabelSelector((id, scoreList)):
    maxScore = float('-inf')
    label = None
    for score in scoreList:
        if score[-1] > maxScore:
            maxScore = score[-1]
            label = score[1]
    return (id, label, maxScore)


def idWordProbGetter(tup1, tup2):
    '''
    Args: ((word, id), unique_w), (word, label, count1, count2, stat, sum_unique_word) or None
    :return: (id, word, probablity)
    '''
    word, id, unique_w = tup1
    if tup2 == None:
        return (id, word, Math.log(1 / float(unique_w)))


def map_word_counter((label_w,count)):
    label, word = label_w.split(',W=')
    label = label.split('Y=')[-1]
    return (word, label, count)


def map_label_Prob((((tup1, tup2), unique_word),label_count_list)):
    '''
    :param tup1: (label, id)
    :param tup2: (label, listof(word,label, count))
    :param unique_word: count number of the unique words
    :param label_count_list: list of star word count for each label
    :yield: (label, id, prob)
    '''
    word, _id = tup1
    dic = {}
    if tup2 == None:
        for k,v in label_count_list:
            yield (k, _id, math.log(1/float(v + unique_word)))
    else:
        word2, label_list = tup2
        for _w, label, count in label_list:
            dic[label] = count
        for k,v in label_count_list:
            if k in dic:
                yield (k, _id, math.log((dic[k] + 1)/float(v + unique_word)))
            else:
                yield (k, _id, math.log(1/float(v + unique_word)))


class NB(Planner):
    # params is a dictionary of params given on the command line.
    # e.g. trainFile = params['trainFile']
    params = GPig.getArgvParams()
    #--params key:value
    LabelsDoc = ReadLines(params.get('trainFile')) | FlatMap(by=labelDocGen)
    #generate test data----(word,id)
    queryWordId = ReadLines(params.get('testFile')) | FlatMap(by=WordIdGen)

    data = FlatMap(LabelsDoc, by=lambda (label, doc): labelTokenGen((label, doc)))

    #count the label occurance
    nlabel_star = Map(LabelsDoc, by=lambda (label, doc): 'Y=*') | Group(by=lambda x:x, reducingTo=ReduceToCount())
    labels = Map(LabelsDoc, by=lambda (label, doc): 'Y=%s'%label) | Group(by=lambda x:x, reducingTo=ReduceToCount())
    #number of unique label occurs
    nlabel = Group(labels, by=lambda (label, count):'ANY', reducingTo=ReduceTo(int, lambda accum,count:accum+1)) | \
             ReplaceEach(by=lambda (dummy, n): n)

    #count the token occurance 'Y=%s,W=*'
    label_anyWord = Map(data, by=lambda (label, word): label) | \
                    Group(by=lambda x:x,  reducingTo=ReduceToCount()) | \
                    Group(by=lambda x:'ANY') | \
                    ReplaceEach(by=lambda (dummy,count_list):count_list)
    label_word = Map(data, by=lambda (label, word): 'Y=%s,W=%s' %(label, word)) | \
                 Group(by=lambda x:x, reducingTo=ReduceToCount())
    unique_word = Distinct(Map(data, by=lambda (label, word): word)) | \
                  Group(by=lambda x:'ANY', reducingTo=ReduceTo(float,lambda accum,count:accum+1)) | \
                  ReplaceEach(by=lambda (dummy, n): n)

    word_counter = Map(label_word, by= map_word_counter) | \
                   Group(by=lambda (word, label, count): word)
    cmp1 = Join(Jin(queryWordId, by= lambda (word, id):word, outer=True), Jin(word_counter, by=lambda (word,label_list):word))
    cmp2 = Augment(cmp1, sideview=unique_word, loadedBy=lambda v: GPig.onlyRowOf(v))
    cmp3 = Augment(cmp2, sideview=label_anyWord, loadedBy=lambda v: GPig.onlyRowOf(v))
    cmp4 = FlatMap(cmp3, by=map_label_Prob) | \
           Group(by=lambda (label, id, p):(label, id), reducingTo=ReduceTo(float,lambda accum,(label, id, p):accum+p))
    labelStat = Join(Jin(nlabel_star, by=lambda (star, sum):'ANY'), Jin(nlabel, by=lambda (count):'ANY')) | \
                ReplaceEach(by=lambda ((star,Nsum),Kinds): ('*',Nsum+Kinds)) | \
                JoinTo(Jin(labels, by=lambda (label, count):'*'), by=lambda (star, n):'*') | \
                ReplaceEach(by=lambda ((star, divider),(label, count)):(label, math.log(int(count+1)/float(divider))))
    cmp5 = Join(Jin(cmp4, by=lambda((label,_id), score):label), Jin(labelStat, by=lambda(label, score):label.split('Y=')[-1])) |\
           ReplaceEach(by=lambda (((label,_id),score1), (label2, score2)):(_id, label, score1 + score2))
    output = Group(cmp5, by=lambda (_id, label, score): (_id)) | \
             Map(by=bestLabelSelector)
# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here