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

class RELU:
    def __init__(self):
        self.type="RELU";
    def eval(self,array):
        return np.maximum(array,0);

class Linear:
    def __init__(self):
        self.type="Linear";
    def eval(self,array):
        return array;