import numpy as np;

from layers.units import unit_factory;
from abc import ABCMeta,abstractmethod;

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def _init_units(self,*args,**kwargs):
        pass;
    @abstractmethod
    def _init_para(self,*args,**kwargs):
        pass;
    @abstractmethod
    def init_weight(self,*args,**kwargs):
        pass;
    @abstractmethod
    def dump(self):
        pass;

class FullyConnect(Layer):
    ## print information of the layer
    def dump(self):
        pass;
    def _init_units(self,size,typ):
        self.units = [];
        for i in range (1,size):
            self.units.append(unit_factory(typ));
    def _init_bias(self,size,typ):
        # init bias as size*1 vector with initial value of 0.1
        self.bias = np.ones((size,1)) * 0.1;
    def _init_para(self,size,typ,activation):
        self.size = size;
        # will be assign in network initialization
        self.previous_size = 0;
        self.typ = typ;
        self.activation = activation;
        self._init_units(size,self.activation);
        self._init_bias(size,self.activation);
    def init_weight(self,previous_size):
        if previous_size is 0:
            return;
        # 2d vector (size * previous_size)
        weight = np.zeros((previous_size, self.size));
        # matrix (size * previous_size)
        self.weight = np.asmatrix(weight);
    def __init__(self,size,activation="RELU"):
        self._init_para(size,"fully_connected",activation);

class Input(Layer):
    ## print the information of the layer
    def dump(self):
        pass;
    def _init_units(self,size,typ):
        self.units = [];
        for i in range (1,size):
            self.units.append(unit_factory(typ))
    def init_weight(self,*args,**kwargs):
        pass;
    def _init_para(self,size):
        self.size = size;
        self.previous_size = 0;
        self.typ = "fully_connected";
        self.activation = "linear";
        self._init_units(size,self.activation);
    def __init__(self,size):
        self._init_para(size);

    
