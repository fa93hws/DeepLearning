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
        raise Exception("Cost function type not found",
                        "There is no cost function type name {0}".format(options.typ));

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
    def get_gradient(self,layers,x,predict,accurate):
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
    def get_value(self,predict,accurate):
        pass;
    def get_gradient(self,layers,x,predict,accurate):
        pass;

class L2Dist(CostFun):
    def dump(self):
        super(CrossEntropy, self).dump();
    def __init__(self,options):
        super(L2Dist, self).__init__(options);
    def get_value(self,predict,accurate):
        # input,accurate: np matrix, 1 * n_outputs
        # input,predict: np matrix, 1 * n_outputs
        # output, scalr
        diff = predict - accurate;
        l2norm = np.linalg.norm(diff);
        return l2norm **2 / 2;
    def get_gradient(self,predict,accurate):
        # input,accurate: np matrix, 1 * n_outputs
        # input,predict: np matrix, 1 * n_outputs
        # output, np matrix 1 * n_outputs
        diff = predict - accurate;
        return diff;