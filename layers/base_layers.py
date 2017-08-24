from layers.units import unit_factory;
from abc import ABCMeta,abstractmethod;

class Layer(metaclass=ABCMeta):
    def _init_units(self,size,typ):
        self.units = [];
        for i in range (1,size):
            self.units.append(unit_factory(typ));
    def _init_weight(self,size,typ):
        self.weight = [];
        for i in range (1,size):
            self.weight.append(0.1);
        self.bias = 0.1;
    def _init_para(self,size,typ,activation):
        self.size = size;
        self.typ = typ;
        self.activation = activation;
        self._init_units(size,self.activation);
        self._init_weight(size,self.activation);

class Input(Layer):
    def __init__(self,size):
        self._init_para(size,"fully_connected","linear");


class Output(Layer):
    def __init__(self,size,activation):
        self._init_para(size,"fully_connected",activation);

class FullyConnect(Layer):
    def __init__(self,size,activation="RELU"):
        self._init_para(size,"fully_connected",activation);
    
