import numpy as np;
from abc import ABCMeta, abstractmethod;

from networks.optimizer import Optimizer;

def solver_factory(options, cost_func):
    # default type is SGD
    if options.get("typ","SGD") is "SGD":
        return SGD(options, cost_func);
    else:
        raise Exception("Unknown solver","{0} is unknown".format(typ));

class Solver(metaclass = ABCMeta):
    @abstractmethod
    def dump(self):
        print("");
        print("Solver type is {0}".format(self.typ));
    @abstractmethod
    def optimize(self,layers,x,y):
        # x,y are numpy tensor
        pass;
    def __init__(self,options,cost_func):
        self.cost_func = cost_func;
        self.typ = options.get("typ","SGD");

class SGD(Solver):
    def dump(self):
        super(SGD, self).dump();
        print("Learning rate is {0}".format(self.alpha));
        print("Batch size is {0}".format(self.batch));
    def __init__(self,options,cost_func):
        super(SGD, self).__init__(options,cost_func);
        self.alpha = options.get("alpha",0.001);
        self.batch = options.get("batch",50);
        self.eps = options.get("eps",1e-6);
        self.max_iter = options.get("max_iter",10**6);
    ## optimize
    def optimize(self,layers,x,y):
        optimizer = Optimizer(layers,x,y,self);