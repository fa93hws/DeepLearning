import numpy as np;

from networks.cost_func import cost_fun_factory;
from networks.solver import solver_factory;

class MLPRegressor:
    ## print details of the network
    def dump(self):
        print("It is a MLP regressor");
        print("It has {0} layers".format(self.num_layers));
        # print("Cost function is {0}".format(self.cost_func.typ));
        for layer in self.layers:
            layer.dump();
        self.solver.dump();
    ## main method of the network
    def _init_layers(self):
        # first input layer do not have weight
        for i in range(1,self.num_layers):
            self.layers[i].init_weight( self.layers[i-1].size );
            self.layers[i].id = i;
            assert self.layers[i].typ == "fully_connected";
    def __init__(self, layers, cost_func_options={}, solver_options={}):
        #### cost function options
        # typ: type of the cost function, default cross_entropy
        # lambda: weight decay factor, default 0
        #### solver options
        # typ: type of the solver, default SGD
        # batch: batch size (for SGD), default 50
        # alpha: learning rate, default 0.001
        # eps: threashold, default 1e-6
        # max_iter: max number of iteration, default 1e6
        self.layers = layers;
        self.num_layers = len(layers);
        self._init_layers();
        self.cost_func = cost_fun_factory(cost_func_options);
        self.solver = solver_factory(solver_options, self.cost_func);
    def train(self,x,y):
        # input,x: a list of training input
        # input,y: a list of expected result
        # 1st step: transform inputs to
        # temp,x: a list of np 2d array,based on row (if input is a list)
        # temp,y: a list of np row array (if input is a list)
        x = [ np.asarray([a]) for a in x ];
        y = [ np.asarray(a) for a in y ];
        self.solver.optimize(self.layers,x,y);    
    def predict(self,x):
        for layer in self.layers:
            if layer.typ is "input":
                continue;
            temp_out = layer.eval(x);
        predict = temp_out;
        return predict;