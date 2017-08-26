from abc import ABCMeta,abstractmethod;
import numpy as np;

def unit_factory(typ):
    if typ is "linear":
        return Linear();
    elif typ is "RELU":
        return RELU();
    else:
        print("No unit named {0}".format(typ));

class Unit(metaclass = ABCMeta):
    @abstractmethod
    def eval(self,array):
        pass;
    @abstractmethod
    def eval_deri(self,array):
        pass;
    @abstractmethod
    def __init__(self,typ):
        self.type = typ;

class RELU(Unit):
    def __init__(self):
        super(RELU, self).__init__("RELU");
    def eval(self,array):
        return np.maximum(array,0);
    def eval_deri(self,array):
        deri = np.ones(array.shape);
        deri[array<0] = 0;

class Linear(Unit):
    def __init__(self):
        super(Linear, self).__init__("linear");
    def eval(self,array):
        return array;
    def eval_deri(self,array):
        return np.ones(array.shape);