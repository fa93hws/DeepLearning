import numpy as np;

from layers.units import unit_factory;
from abc import ABCMeta,abstractmethod;

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def _init_para(self):
        pass;
    @abstractmethod
    def init_weight(self,*args,**kwargs):
        pass;
    @abstractmethod
    def dump(self):
        print("");
        print("layer {0} is a/an {1} layer".format(self.id, self.typ));
        print("It has {0} {1} units".format(self.size, self.activation));
    def __init__(self,size,typ,activation):
        self.size = size;
        # will be assigned in network
        self.id = 0;
        # name of the layers
        self.typ = typ;
        self.activation = activation;
        self._init_units();
        self._init_para();
    def _init_units(self):
        self.units = [];
        for i in range (1,self.size):
            self.units.append(unit_factory(self.activation));

class FullyConnect(Layer):
    ## print information of the layer
    def dump(self):
        super(FullyConnect, self).dump();
        print("weight is {0}".format(self.weight));
        print("bias is {0}".format(self.bias));
    ## main methods
    def _init_bias(self):
        # init bias as size*1 vector with initial value of 0.1
        self.bias = np.ones((self.size,1)) * 0.1;
    def _init_para(self):
        # will be assign in network initialization
        self.previous_size = 0;
        self._init_bias();
    def init_weight(self,previous_size):
        if previous_size is 0:
            return;
        # 2d vector (size * previous_size)
        weight = np.zeros((previous_size, self.size));
        # matrix (size * previous_size)
        self.weight = np.asmatrix(weight);
    def __init__(self,size,activation="RELU"):
        self.activation = activation;
        super(FullyConnect,self).__init__(size,"fully_connected",activation);
        # self._init_para(size,"fully_connected",activation);

class Input(Layer):
    ## print the information of the layer
    def dump(self):
        super(Input, self).dump();
    ## main methods
    def init_weight(self,*args,**kwargs):
        pass;
    def _init_para(self):
        pass;
    def __init__(self,size):
        super(Input,self).__init__(size,"input","linear");

    
