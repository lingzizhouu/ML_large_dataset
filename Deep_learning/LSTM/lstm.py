"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import time
import numpy as np
from xman import *
from utils import *
from autograd import *

#np.random.seed(0)

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.max_len = max_len
        self.in_size = in_size
        self.num_hid = num_hid
        self.out_size = out_size
        self.my_xman = self._build() #DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()

        x.y = f.input(name='y', default=np.zeros((1, self.out_size)))
        x.h0 = f.input(name='h0', default=np.zeros((1, self.num_hid)))
        x.c0 = f.input(name='c0', default=np.zeros((1, self.num_hid)))

        a_W = np.sqrt(6/float(self.in_size + self.num_hid))
        a_b = np.sqrt(6/float(self.num_hid))
        a_U = np.sqrt(6/float(self.num_hid + self.num_hid))
        x.Wi = f.param(name='Wi', default=np.random.uniform(-a_W, a_W, (self.in_size, self.num_hid)))
        x.Wf = f.param(name='Wf', default=np.random.uniform(-a_W, a_W, (self.in_size, self.num_hid)))
        x.Wo = f.param(name='Wo', default=np.random.uniform(-a_W, a_W, (self.in_size, self.num_hid)))
        x.Wc = f.param(name='Wc', default=np.random.uniform(-a_W, a_W, (self.in_size, self.num_hid)))
        x.bi = f.param(name='bi', default=0.1 * np.random.uniform(-a_b, -a_b, (self.num_hid,)))
        x.bf = f.param(name='bf', default=0.1 * np.random.uniform(-a_b, a_b, (self.num_hid,)))
        x.bo = f.param(name='bo', default=0.1 * np.random.uniform(-a_b, a_b, (self.num_hid,)))
        x.bc = f.param(name='bc', default=0.1 * np.random.uniform(-a_b, a_b, (self.num_hid,)))
        x.Ui = f.param(name='Ui', default=np.random.uniform(-a_U, a_U, (self.num_hid, self.num_hid)))
        x.Uf = f.param(name='Uf', default=np.random.uniform(-a_U, a_U, (self.num_hid, self.num_hid)))
        x.Uo = f.param(name='Uo', default=np.random.uniform(-a_U, a_U, (self.num_hid, self.num_hid)))
        x.Uc = f.param(name='Uc', default=np.random.uniform(-a_U, a_U, (self.num_hid, self.num_hid)))

        for i in range(1, self.max_len + 1):
            setattr(x, 'x' + str(i), f.input(name='x%s' % i, default=np.random.rand(1, self.in_size)))

            setattr(x, 'i' + str(i), f.sigmoid(f.mul(getattr(x, 'x%s' % i), x.Wi) + f.mul(getattr(x, 'h%s' % (i - 1)), x.Ui) + x.bi))
            setattr(x, 'f' + str(i), f.sigmoid(f.mul(getattr(x, 'x%s' % i), x.Wf) + f.mul(getattr(x, 'h%s' % (i - 1)), x.Uf) + x.bf))
            setattr(x, 'o' + str(i), f.sigmoid(f.mul(getattr(x, 'x%s' % i), x.Wo) + f.mul(getattr(x, 'h%s' % (i - 1)), x.Uo) + x.bo))
            setattr(x, 'c_' + str(i), f.tanh(f.mul(getattr(x, 'x%s' % i), x.Wc) + f.mul(getattr(x, 'h%s' % (i - 1)), x.Uc) + x.bc))

            setattr(x, 'c' + str(i), f.hadamard(getattr(x, 'f%s' % i), getattr(x, 'c%s' % (i - 1))) + f.hadamard(getattr(x, 'i%s' % i), getattr(x, 'c_%s' % i)))
            setattr(x, 'h' + str(i), f.hadamard(getattr(x, 'o%s' % i), f.tanh(getattr(x, 'c%s' % i))))

            #x.it = f.sigmoid(f.mul(self.input['x%s' % i], x.Wi) + f.mul(x.h, x.Ui) + x.bi)
            #x.ft = f.sigmoid(f.mul(self.input['x%s' % i], x.Wf) + f.mul(x.h, x.Uf) + x.bf)
            #x.ot = f.sigmoid(f.mul(self.input['x%s' % i], x.Wt) + f.mul(x.h, x.Ui) + x.bi)
            #x.ct_ = f.tanh(f.mul(self.input['x%s' % i], x.Wt) + f.mul(x.h, x.Uf) + x.bf)
            #x.ct = f.hadamard(x.ft, x.c0) + f.hadamard(x.it, x.ct_)
            #x.ht = f.hadamard(x.ot, f.tanh(x.ct))

        a_W2 = np.sqrt(6 / float(self.num_hid + self.out_size))
        a_b2 = np.sqrt(6 / float(self.out_size))

        x.W2 = f.param(name='W2', default=np.random.uniform(-a_W2, a_W2, (self.num_hid, self.out_size)))
        x.b2 = f.param(name='b2', default=0.1 * np.random.uniform(-a_b2, a_b2, (self.out_size,)))

        x.O2 = f.relu(f.mul(getattr(x, 'h%s' % i), x.W2) + x.b2)
        x.outputs = f.softmax(x.O2)
        x.loss = f.mean(f.crossEnt(x.outputs, x.y))

        return x.setup()

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    train_loss_file = params['train_loss_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    #OPTIONAL: CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    my_xman = lstm.my_xman
    value_dict = my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    ad = Autograd(my_xman)
    # initialization
    wengert_list = my_xman.operationSequence(my_xman.loss)
    min_validation_loss = sys.maxint
    min_loss_value_dict = dict


    print 'Wengart list:'
    for (dstVarName, operator, inputVarNames) in wengert_list:
        print '  %s = %s(%s)' % (dstVarName, operator, (",".join(inputVarNames)))
    training_time = 0

    for i in range(epochs):
        t_train_start = time.time()
        eval_count = 0
        bprop_count = 0
        value_dict['eval'] = 0
        value_dict['bprop'] = 0
        for (idxs,e,l) in mb_train:

            # TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            # valueDict contains current value of params
            # xc and yc are inputs for current minibatch
            # e has shape of N*M*V, while we require stream of N*V of size M
            for idx in range(max_len):
                # value_dict['x%s' % (idx + 1)] = e[:, idx, :]
                value_dict['x%s' % (idx + 1)] = e[:, max_len - idx - 1, :]

            value_dict['y'] = l
            value_dict['h0'] = np.zeros((e.shape[0], num_hid))
            value_dict['c0'] = np.zeros((e.shape[0], num_hid))
            value_dict = ad.eval(wengert_list, value_dict)  # forward pass
            gradientDict = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))  # backward pass


            # update the parardmeters appropriately
            for rname in gradientDict:
                if my_xman.isParam(rname):
                    value_dict[rname] = value_dict[rname] - lr * gradientDict[rname]

            #save the train loss
            train_loss = np.append(train_loss, value_dict['loss'])
            #print "training loss is {0}".format(value_dict['loss'])
        print "train_loss is {0}".format(np.sum(train_loss, axis=0))
        print 'eval takes {0} times, bprop takes {1} times'.format(value_dict['eval'], value_dict['bprop'])
        # sum the training time
        training_time += time.time() - t_train_start
        # validate
        validate_loss = np.ndarray([0])
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
            for idx in range(max_len):
                value_dict['x%s' % (idx + 1)] = e[:, max_len - idx - 1, :]

            value_dict['y'] = l
            value_dict['h0'] = np.zeros((e.shape[0], num_hid))
            value_dict['c0'] = np.zeros((e.shape[0], num_hid))

            value_dict = ad.eval(wengert_list, value_dict)  # forward pass
            validate_loss = np.append(validate_loss, value_dict['loss'])



        #TODO compare current validation loss to minimum validation loss
        sum_loss = np.sum(validate_loss, axis=0)
        if sum_loss < min_validation_loss:
            min_validation_loss = sum_loss
            min_loss_value_dict = dict(value_dict)
        # and store params if needed
    print "done"
    #write out the train loss

    np.save(train_loss_file, train_loss)
    ouput_probabilities = np.ndarray([0])
    value_dict = min_loss_value_dict
    test_loss = np.ndarray([0])
    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        for idx in range(max_len):
            value_dict['x%s' % (idx + 1)] = e[:, max_len - idx - 1, :]

        value_dict['y'] = l
        value_dict['h0'] = np.zeros((e.shape[0], num_hid))
        value_dict['c0'] = np.zeros((e.shape[0], num_hid))
        value_dict = ad.eval(wengert_list, value_dict)  # forward pass
        test_loss = np.append(test_loss, value_dict['loss'])

        ouput_probabilities = np.append(ouput_probabilities, value_dict['outputs'])

    print test_loss
        
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    ouput_probabilities = ouput_probabilities.reshape(ouput_probabilities.shape[0] / mb_train.num_labels,
                                                      mb_train.num_labels)
    np.save(output_file, ouput_probabilities)

    print "batch_size is {0},{1}epoch takes {2}, {3}on average".format(batch_size, epochs, training_time,
                                                                       training_time / epochs)
    print value_dict['eval']
    print value_dict['bprop']

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    parser.add_argument('--train_loss_file', dest='train_loss_file', type=str, default='train_loss')
    params = vars(parser.parse_args())
    main(params)
