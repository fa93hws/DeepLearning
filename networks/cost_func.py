import math;
from abc import ABCMeta,abstractmethod;
import numpy as np;

def cost_fun_factory(options):
    # default value of typ is cross_entropy
    if options.get("typ","cross_entropy") is "cross_entropy":
        return CrossEntropy(options);
    elif options["typ"] is "l2norm":
        return L2Dist(options);
    else:
        raise Exception("Cost function type not found","There is no cost function type name {0}".format(options.typ));

class CostFun(metaclass=ABCMeta):
    @abstractmethod
    def dump(self):
        print("Type of cost function is {0}".format(self.typ));
        print("Weight decay factor is {0}".format(self.decay));
    @abstractmethod
    def get_value(self,predict,accurate):
        # predict: a list of np row array
        # accurate: a list of np row array
        pass;
    @abstractmethod
    def get_deri(self,layers,x,predict,accurate):
        # input is numpy array
        pass;
    def __init__(self,options):
        self.decay = options.get("lambda",0);
        self.typ = options.get("typ","cross_entropy");


class CrossEntropy(CostFun):
    def dump(self):
        super(CrossEntropy, self).dump();
    def __init__(self,options):
        super(CrossEntropy, self).__init__(options);
    # def getValue(self,x,y):
    #     pass;
    # def getDeri(self,x,y):
    #     pass;

class L2Dist(CostFun):
    def dump(self):
        super(CrossEntropy, self).dump();
    def __init__(self,options):
        super(L2Dist, self).__init__(options);
    def get_value(self,predict,accurate):
        # predict: a list of np row array
        # accurate: a list of np row array
        cost = 0.0;
        len = 0;
        for p,a in zip(predict,accurate):
            len += 1;
            cost += np.linalg.norm(p-a);
        return cost/len/2;
    def _get_deri_w():
        pass;
    def _get_deri_b():
        pass;
    def get_deri(self,layers,x,predict,accurate):
        # store deri for every layer in chain
        # first value in chain -> deri in last layer
        chain = [];
        n_layers = len(layers);
        for i in range(n_layers-1,0):
            print(i)
        return dw,db;