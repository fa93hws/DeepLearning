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
    def train(self,li_x,li_y):
        # input,li_x: a list of training input
        # input,li_y: a list of expected result
        # 1st step: transform inputs to
        # temp,x: np matrix, n_samples * n_features
        # temp,y: np matrix, n_samples * n_outputs
        n_samples = len(li_x);
        n_features = len(li_x[0]);
        n_outputs = 1;
        x = np.asmatrix( np.zeros(( n_samples,n_features)) );
        y = np.asmatrix( np.zeros(( n_samples, n_outputs)) );
        for i in range(0,n_samples):
            x[i,:] = np.asarray(li_x[i]);
            y[i,:] = np.asarray(li_y[i]);
        self.solver.optimize(self.layers,x,y);    
    def predict(self,x):
        # input,x: a list of list of features (n_sample * n_features)
        # output, predict: a 2d np array (n_sample * n_result)
        # mid,x: np matrix n_sample * n_features
        x = np.asmatrix(x);
        n_sample = x.shape[0];
        n_result = self.layers[-1].size;
        predict = np.zeros((n_sample,n_result));
        for i in range(0,n_sample):
            temp_out = x[i,:].T;
            for layer in self.layers:
                temp_out = layer.eval(temp_out);
            predict[i,:] = temp_out;

        return predict;