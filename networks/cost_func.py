import math;
from abc import ABCMeta,abstractmethod;
import numpy as np;

class CostFun(metaclass=ABCMeta):
    pass;
    # @abstractmethod
    # def getValue(self,x,y):
    #     pass;
    # @abstractmethod
    # def getDeri(self,x,y):
    #     pass;


class CrossEntropy(CostFun):
    pass;
    # def getValue(self,x,y):
    #     pass;
    # def getDeri(self,x,y):
    #     pass;

class L2Dist(CostFun):
    def getValue(self,x,y):
        x = np.asarray();
        y = np.asarray();
        diff = x.substract(y);
        return np.linalg.norm(diff);
    def getDeri(self,x,y):
        x = np.asarray();
        y = np.asarray();
        