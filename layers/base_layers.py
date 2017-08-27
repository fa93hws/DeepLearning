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
    @abstractmethod
    def eval(self,input):
        pass;
    def __init__(self,size,typ,activation):
        assert size > 0;
        self.size = size;
        # will be assigned in network
        self.id = 0;
        # name of the layers
        self.typ = typ;
        self.activation = activation;
        self._init_unit();
        self._init_para();
    def _init_unit(self):
        self.unit = unit_factory(self.activation);

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
        super(FullyConnect,self).__init__(size,"fully_connected",activation);
    def eval(self,input):
        # input : np array (n*1)
        # output: np array (m*1)
        if (input.shape[0] is not self.size):
            raise Exception("size mismatch"
                            ,"input in the fully connected layer({0}) should be equal to the number of neurals({1})"
                            .format(input.shape[0], self.size));
        raw = self.weight * input + self.bias;
        out = self.unit.eval(raw);
        return out;

class Input(Layer):
    ## print the information of the layer
    def dump(self):
        super(Input, self).dump();
    ## main methods
    def init_weight(self,*args,**kwargs):
        pass;
    def _init_para(self):
        pass;
    def eval(self,input):
        # input : np array (n*1)
        # output: np array (m*1)
        return input;
    def __init__(self,size):
        super(Input,self).__init__(size,"input","linear");