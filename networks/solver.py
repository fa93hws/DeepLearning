import numpy as np;
from random import randrange;
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
    def _fill_batch(self,x,y):
        # input,x: np matrix, n_samples * n_features
        # input,y: np matrix, n_samples * n_outputs
        # output,batch_x: np matrix, batch_size * n_features
        # output,batch_y: np matrix, batch_size * n_output
        n_features = x.shape[1];
        n_outputs = y.shape[1];
        batch_x = np.zeros((self.batch,n_features));
        batch_y = np.zeros((self.batch,n_outputs));
        size = len(x);
        # can be optimized via using multi-threading
        for i in range (0, self.batch):
            idx = randrange(0,size);
            batch_x[i,:] = x[idx,:];
            batch_y[i,:] = y[idx,:];
        return batch_x, batch_y;
    def optimize(self,layers,x,y):
        # input,x: np matrix, n_samples * n_features
        # input,y: np matrix, n_samples * n_outputs
        optimizer = Optimizer(self.cost_func, self.alpha, self.eps);
        for i in range(1, self.max_iter):
            batch_x, batch_y = self._fill_batch(x,y);
            finish_flag = optimizer.do_once(layers,batch_x,batch_y);
            break;