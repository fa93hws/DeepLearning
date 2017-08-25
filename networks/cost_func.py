import math;
from abc import ABCMeta,abstractmethod;

def cost_fun_factory(typ):
    if typ is "cross_entropy":
        return CrossEntropy();
    elif typ is "l2norm":
        return L2Dist();

class CostFun(metaclass=ABCMeta):
    pass;
    # @abstractmethod
    # def get_value(self,x,w,b,y):
    # input is numpy array
    #     pass;
    # @abstractmethod
    # def get_deri(self,x,w,b,y):
    #     pass;


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
        dist = 0.0;
        for p,a in zip(predict,accurate):
            dist += (p-a) ** 2;
        return math.sqrt(dist);
    def get_deri(self,x,w,b,y):
        pass;