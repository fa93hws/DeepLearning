import numpy as np;
from abc import ABCMeta, abstractmethod;

from networks.optimizer import Optimizer;

def solver_factory(typ,cost_func,options):
    if typ is "SGD":
        return SGD(cost_func,options);
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
    def __init__(self,cost_func,typ):
        self.cost_func = cost_func;
        self.typ = typ;

class SGD(Solver):
    def dump(self):
        super(SGD, self).dump();
        print("Learning rate is {0}".format(self.alpha));
        print("Batch size is {0}".format(self.batch));
    def __init__(self,cost_func,options):
        super(SGD, self).__init__(cost_func,"SGD");
        self.alpha = options["alpha"];
        self.batch = options["batch"];
        self.eps = options["eps"];
        self.max_iter = options["max_iter"];
    ## optimize
    def optimize(self,layers,x,y):
        optimizer = Optimizer(layers,x,y,self);