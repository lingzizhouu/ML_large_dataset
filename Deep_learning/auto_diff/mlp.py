"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
import time
from xman import *
from utils import *
from autograd import *


np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.my_xman = self._build() # DO NOT REMOVE THIS LINE. Store the output of xman.setup() in this variable


    def _build(self):
        x = XMan()
        #TODO define your model here
        x.x = f.input(name='x', default=np.random.rand(1, self.layer_sizes[0]))
        x.y = f.input(name='y', default=np.random.rand(1, self.layer_sizes[2]))
        a_W = np.sqrt(6/float(self.layer_sizes[0] + self.layer_sizes[1]))
        a_W2 = np.sqrt(6/float(self.layer_sizes[1] + self.layer_sizes[2]))
        a_b = np.sqrt(6/float(self.layer_sizes[1]))
        a_b2 = np.sqrt(6 / float(self.layer_sizes[2]))
        x.W = f.param(name='W', default = np.random.uniform(-a_W, a_W, (self.layer_sizes[0], self.layer_sizes[1])))
        x.W2 = f.param(name='W2', default= np.random.uniform(-a_W2, a_W2, (self.layer_sizes[1], self.layer_sizes[2])))
        x.b = f.param(name='b', default = 0.1 * np.random.uniform(-a_b, a_b, (self.layer_sizes[1],)))
        x.b2 = f.param(name='b2', default=0.1 * np.random.uniform(-a_b2, a_b2, (self.layer_sizes[2],)))

        x.o1 = f.relu(f.mul(x.x, x.W) + x.b)
        x.o2 = f.relu(f.mul(x.o1, x.W2) + x.b2)
        x.P = f.softmax(x.o2)
        x.loss = f.mean(f.crossEnt(x.P, x.y))

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
    print mb_train
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    #TODO CHECK GRADIENTS HERE


    print "done"

    # train
    print "training..."
    # get default data and params
    my_xman = mlp.my_xman
    value_dict = my_xman.inputDict()
    lr = init_lr
    train_loss = np.ndarray([0])
    min_validate_loss = np.ndarray([0])
    best_param = dict
    opseq = mlp.my_xman.operationSequence(my_xman.loss)
    print 'Wengart list:'
    for (dstVarName, operator, inputVarNames) in opseq:
        print '  %s = %s(%s)' % (dstVarName, operator, (",".join(inputVarNames)))

    wengert_list = my_xman.operationSequence(my_xman.loss)
    ad = Autograd(my_xman)

    training_time = 0
    for i in range(epochs):
        t_train_start = time.time()
        for (idxs,e,l) in mb_train:
            #TODO prepare the input and do a fwd-bckwd pass over it and update the weights
            # valueDict contains current value of params
            # xc and yc are inputs for current minibatch

            value_dict['x'] = e.reshape([e.shape[0], e.shape[1]*e.shape[2]])
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list, value_dict)  # forward pass
            gradientDict = ad.bprop(wengert_list, value_dict, loss=np.float_(1.))  # backward pass

            # update the parameters appropriately
            for rname in gradientDict:
                if my_xman.isParam(rname):
                    value_dict[rname] = value_dict[rname] - lr * gradientDict[rname]
            #save the train loss
            #print value_dict['loss']
            train_loss = np.append(train_loss, value_dict['loss'])
        #sum the training time
        training_time += time.time() - t_train_start

        print "train_loss is {0}".format(np.sum(train_loss, axis=0))
        #train_loss = np.ndarray([0])

        # validate
        validate_loss = np.ndarray([0])
        for (idxs,e,l) in mb_valid:
            #TODO prepare the input and do a fwd pass over it to compute the loss
            value_dict['x'] = e.reshape([e.shape[0], e.shape[1] * e.shape[2]])
            value_dict['y'] = l
            value_dict = ad.eval(wengert_list, value_dict)  # forward pass
            validate_loss = np.append(validate_loss, value_dict['loss'])

        #TODO compare current validation loss to minimum validation loss
        sum_loss = np.sum(validate_loss, axis=0)
        if min_validate_loss:
            if min_validate_loss > sum_loss:
                min_validate_loss = sum_loss
                best_param = dict(value_dict)

        # and store params if needed

    print "done"
    print "batch_size is {0},{1}epoch takes {2}, {3}on average".format(batch_size, epochs, training_time, training_time / epochs)
    #write out the train loss
    np.save(train_loss_file, train_loss)
    ouput_probabilities = np.ndarray([0])
    for (idxs,e,l) in mb_test:
        value_dict['x'] = e.reshape([e.shape[0], e.shape[1] * e.shape[2]])
        value_dict['y'] = l
        value_dict = ad.eval(wengert_list, value_dict)  # forward pass
        ouput_probabilities = np.append(ouput_probabilities, value_dict['P'])
        # prepare input and do a fwd pass over it to compute the output probs
        
    #TODO save probabilities on test set
    # ensure that these are in the same order as the test input
    print "ouput_probabilities shape is {0}".format(ouput_probabilities.shape)
    print ouput_probabilities.shape[0]
    ouput_probabilities = ouput_probabilities.reshape(ouput_probabilities.shape[0] / mb_train.num_labels, mb_train.num_labels)
    np.save(output_file, ouput_probabilities)
    print "changed shape is {0}".format(ouput_probabilities.shape)

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

