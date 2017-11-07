# a sample use of xman for an optimization task
# 

import sys
from xman import *

# functions I'll use for this problem

class f(XManFunctions):
    @staticmethod 
    def half(a): 
        return XManFunctions.registerDefinedByOperator('half',a)
    @staticmethod 
    def square(a): 
        return XManFunctions.registerDefinedByOperator('square',a)
    @staticmethod 
    def alias(a): 
        """ This will just make a copy of a register that
        has a different name."""
        return XManFunctions.registerDefinedByOperator('alias',a)

# the functions used to evaluate each function above, to be called
# with the functions actual inputs as arguments.  'mul', 'add' and
# 'subtract' are built-in in xman.py (and support operator
# overloading)

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'mul':      lambda x1,x2: x1*x2,
    'half':     lambda x: 0.5*x,
    'square':   lambda x: x*x,
    'alias':    lambda x: x,
    }

# the functions that backprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]

BP_FUNS = {
    'add':      [lambda delta,out,x1,x2: delta,    lambda delta,out,x1,x2: delta],
    'subtract': [lambda delta,out,x1,x2: delta,    lambda delta,out,x1,x2: -delta],
    'mul':      [lambda delta,out,x1,x2: delta*x2, lambda delta,out,x1,x2: delta*x1],
    'half':     [lambda delta,out,x: delta*0.5],
    'square':   [lambda delta,out,x: delta*2*x],
    'alias':    [lambda delta,out,x: delta],
    }

class Autograd(object):
    """ Automatically compute partial derivatives.
    """

    def __init__(self,xman):
        self.xman = xman

    def eval(self,opseq,valueDict):
        """ Evaluate the function specified by the wengart list (aka operation
        sequence). Here valueDict is a dict holding the values of any
        inputs/parameters that are needed (indexed by register name,
        which is a string).  Returns the augmented valueDict.
        """
        for (dstName,funName,inputNames) in opseq:
            inputValues = map(lambda a:valueDict[a], inputNames)
            fun = EVAL_FUNS[funName] 
            result = fun(*inputValues)
            valueDict[dstName] = result
        return valueDict

    def bprop(self,opseq,valueDict,**deltaDict):
        """ For each intermediate register g used in computing the function f
        computed by the operation sequence, find df/dg.  Here
        valueDict is a dict holding the values of any
        inputs/parameters that are needed for the gradient (indexed by
        register name), as populated by eval.
        """
        for (dstName,funName,inputNames) in reversed(opseq):
            delta = deltaDict[dstName]
            # values is extended to include the next-level delta and
            # the function output, and these will be passed as
            # arguments
            values = [delta] + map(lambda a:valueDict[a], [dstName]+list(inputNames))
            for i in range(len(inputNames)):
                result = (BP_FUNS[funName][i])(*values)
                # increment a running sum of all the delta's that are
                # pushed back to the i-th parameter, initializing to
                # zero if needed.
                self._incrementBy(deltaDict, inputNames[i], result)
        return deltaDict

    def _incrementBy(self, dict, key, inc):
        if key not in dict: dict[key] = 0
        dict[key] = dict[key] + inc

#
# a simple test case
#


def House():
    """ Expression manager for a toy task that has parameters and a loss function.
    First you compute the area of a simple shape, a 'house',
    which is a triangle on top of a rectangle.
    """ 

    # define some macros
    def roofHeight(wallHeight): 
        return f.half(wallHeight)
    def triangleArea(h,w): 
        return f.half(h*w)
    def rectArea(h,w): 
        return h*w

    #create the instance
    x = XMan()

    # height and width of rectangle are inputs
    x.h = f.param(default=30.0)
    print "x.h is {0}".format(x.h)
    x.w = f.param(default=20.0)
    # so is the target height and the target area, these inputs have
    # defaults
    x.targetArea = f.input(default=0.0)
    x.targetHeight = f.input(default=8.0)
    x.heightFactor = f.input(default=100.0)

    # compute area of the house
    x.area = rectArea(x.h,x.w) + triangleArea(roofHeight(x.h), x.w)

    # loss to optimize is weighted sum of square loss of area relative
    # to the targetArea, plus same for height
    x.loss = f.square(x.area - x.targetArea) + f.square(x.h - x.targetHeight) * x.heightFactor

    return x

def Skyscraper(numFloors):
    """ Another toy task - optimize area of a stack of several rectangles
    """ 

    x = XMan()
    # height and width of rectangle are inputs
    x.h = f.param(default=30.0)
    x.w = f.param(default=20.0)
    x.targetArea = f.input(default=0.0)
    x.targetHeight = f.input(default=8.0)
    x.heightFactor = f.input(default=100.0)

    # compute area of the skyscraper
    x.zero = f.input(default=0.0)

    # here areaRegister popints to a different
    # register in each iteration of the loop
    areaRegister = x.zero
    for i in range(numFloors):
        floorRegister = (x.h * x.w)
        floorRegister.name = 'floor_%d' % (i+1)
        areaRegister = areaRegister + floorRegister
    # when the loop finishes, we give it the register a name, in this
    # case by having an instance variable point to the register.
    # we could also execute: areaRegister.name = 'area'
    x.area = areaRegister

    x.loss = f.square(x.area - x.targetArea) + f.square(x.h - x.targetHeight) * x.heightFactor
    return x

if __name__ == "__main__":

    # pick a task from common line args
    if len(sys.argv)>1 and sys.argv[1]=='skyscraper':
        h = Skyscraper(10).setup()
        targetArea=1500
        rate = 0.00001  #slow down the learning rate
    else:
        h = House().setup()
        targetArea = 200
        rate = 0.001

    
    # build your dream house with gradient descent
    autograd = Autograd(h)
    epochs = 20

    # this fills in default values where they exist
    initDict = h.inputDict(h=5,w=10,targetArea=targetArea)

    # form wengart list to compute the loss, and print it
    opseq = h.operationSequence(h.loss)
    print 'Wengart list:'
    for (dstVarName, operator, inputVarNames) in opseq:
        print '  %s = %s(%s)' % (dstVarName,operator,(",".join(inputVarNames)))

    # optimize
    for i in range(epochs):

        # evaluate
        valueDict = autograd.eval(opseq,initDict)

        # display current result
        def displayValues(d,keys): return "\t".join(map(lambda k:'%s=%g' % (k,d[k]), keys.split()))
        print 'epoch',i+1,'\t',displayValues(valueDict,'h w area targetArea loss')
        # stop if converged
        if valueDict['loss'] < 0.01:
            print 'good enough'
            break

        # find gradient of loss wrt parameters
        gradientDict = autograd.bprop(opseq,valueDict,loss=1.0)

        # update the parameters appropriately
        for rname in gradientDict:
            if h.isParam(rname):
                initDict[rname] = initDict[rname] - rate*gradientDict[rname]


