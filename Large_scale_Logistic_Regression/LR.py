import re
import sys
import math
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

def tokenizeDoc(cur_doc):
     return re.findall('\\w+', cur_doc)


def sigmoid(score):
    '''
    :param score: dot product of beta and x;
    :return: special version of sigmoid function to prevent overflow;
    '''
    overflow = 20.0
    if score > overflow:
        score = overflow
    elif score < -overflow:
        score = -overflow
    exp = math.exp(score)
    return exp / (1 + exp)

def hashed_idx(word, vocabulary):
    '''
    :param word: the target word
    :return: hashed index of the word
    '''
    return hash(word) % (vocabulary)


def sgd(a, b, h_tokens, cur_lambda, base, k, y=0):
    '''
    Perform lazy sgd
    :param a: timer dictionary for one of the 5 five classifer
    :param b: param dictionary for one of the 5 five classifer
    :param tokens: the hashed idx of the tokens
    :param cur_lambda: current learning rate
    :param k: number of of training example passed
    :param y: the real answer, 0/1
    :param base: 1 - 2 * cur_lambda * mu
    :param gd: cur_lambda * (y - p)
    :return:
    '''

    #pre-compute the p
    score = 0
    power = 0
    other = 0
    for h_token in h_tokens:
        score += b[h_token]
    p = sigmoid(score)
    gd = cur_lambda * (y - p)

    for h_token in h_tokens:
        b[h_token] *= math.pow(base, k - a[h_token])
        b[h_token] += gd
        a[h_token] = k
    return power,other

def classify(B,C, doc):
    '''
    :param B: learned model dictionary
    :param C: the map of index to label
    :param doc: the corpus
    :return: the classification result
    '''
    ans = ''
    predictions = [0]*5
    for token in tokenizeDoc(doc):
        if token not in stop_word:
            hashed_i = hashed_idx(token, vocabulary)
            for k, v in enumerate(B):
                predictions[k] += v[hashed_i]
    for idx, prediction in enumerate(predictions):
        ans += '%s\t%f,' % (C[idx], sigmoid(prediction))
    return ans[:-1]

k = 0
t = 1
_, vocabulary, _lambda, mu, max_iter, train_size, test_data = sys.argv
vocabulary = int(vocabulary)
_lambda = float(_lambda)
mu = float(mu)
max_iter = int(max_iter)
train_size = int(train_size)
mapidx = 0
# we will have 5 classifier
B = [[0]*vocabulary, [0]*vocabulary, [0]*vocabulary, [0]*vocabulary, [0]*vocabulary]
A = [[0]*vocabulary, [0]*vocabulary, [0]*vocabulary, [0]*vocabulary, [0]*vocabulary]
# map to the index of visited array
C = {}

for line in sys.stdin:
    k = k + 1
    line = line.strip()
    id, labels, doc = line.split("\t")
    to_visit = [True]*5
    cur_lambda = _lambda / t**2
    base = 1 - 2 * cur_lambda * mu
    tokens = [hashed_idx(token, vocabulary) for token in tokenizeDoc(doc) if token not in stop_word]
    for label in labels.split(','):
        if label not in C:
            C[label] = mapidx
            C[mapidx] = label
            mapidx += 1
        sgd(A[C[label]], B[C[label]], tokens, cur_lambda, base, k, y=1)
        to_visit[C[label]] = False
    for idx, not_visited in enumerate(to_visit):
        if not_visited:
            sgd(A[idx], B[idx], tokens, cur_lambda, base, k, y=0)
    if k % train_size == 0:
        t += 1
_lambda /= t**2
for i in xrange(5):
    b = B[i]
    a = A[i]
    for idx in xrange(len(b)):
        b[idx] *= math.pow(1 - 2 * _lambda * mu, k - a[idx])


with open(test_data, 'rb') as tests:
    count = 0
    correct = 0
    for line in tests:
        count += 1
        id, labels, doc = line.split("\t")
        s = set(labels.split(','))
        print classify(B, C, doc)