import math;
from abc import ABCMeta,abstractmethod;
import numpy as np;

def cost_fun_factory(typ):
    if typ is "cross_entropy":
        return CrossEntropy();
    elif typ is "l2norm":
        return L2Dist();

class CostFun(metaclass=ABCMeta):
    pass;
    @abstractmethod
    def get_value(self,predict,accurate):
        # predict: a list of np row array
        # accurate: a list of np row array
        pass;
    @abstractmethod
    def get_deri(self,layers,x,predict,accurate):
        # input is numpy array
        pass;


class CrossEntropy(CostFun):
    def __init__(self):
        self.typ = "cross entropy";
    # def getValue(self,x,y):
    #     pass;
    # def getDeri(self,x,y):
    #     pass;

class L2Dist(CostFun):
    def __init__(self):
        self.typ = "l2norm";
    def get_value(self,predict,accurate):
        # predict: a list of np row array
        # accurate: a list of np row array
        cost = 0.0;
        len = 0;
        for p,a in zip(predict,accurate):
            len += 1;
            cost += np.linalg.norm(p-a);
        return cost/len/2;
    def get_deri(self,layers,x,predict,accurate):
        pass;