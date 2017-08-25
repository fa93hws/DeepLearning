import math;
from abc import ABCMeta,abstractmethod;


from utils.num import to_nparrays;

def cost_fun_factory(typ):
    if typ is "cross_entropy":
        return CrossEntropy();
    elif typ is "l2norm":
        return L2Dist();

class CostFun(metaclass=ABCMeta):
    @abstractmethod
    def getValue(self,x,w,b,y):
        pass;
    @abstractmethod
    def getDeri(self,x,w,b,y):
        pass;


class CrossEntropy(CostFun):
    pass;
    # def getValue(self,x,y):
    #     pass;
    # def getDeri(self,x,y):
    #     pass;

class L2Dist(CostFun):
    def getValue(self,x,w,b,y):
        pass;
    def getDeri(self,x,w,b,y):
        pass;