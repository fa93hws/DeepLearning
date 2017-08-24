from abc import ABCMeta,abstractmethod;

def unit_factory(typ):
    return{
        "linear": Linear(),
        "RELU": RELU(),
    }[typ];

class Unit(metaclass = ABCMeta):
    @abstractmethod
    def get(self,val):
        pass;

class RELU:
    def __init__(self):
        self.type="RELU";
    def get(self,val):
        return max(0,val);

class Linear:
    def __init__(self):
        self.type="Linear";
    def get(self,val):
        return val;