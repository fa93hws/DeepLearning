import numpy as np;

from networks.cost_func import cost_fun_factory;

class MLPRegressor:
    ## print details of the network
    def dump(self):
        print("It is a MLP regressor");
        print("It has {0} layers".format(self.num_layers));
        print("Solver type is {0}".format(self.solver));
        print("Learning rate is {0}".format(self.alpha));
        print("Cost function is {0}".format(self.cost_func.typ));
        print("batch_size is {0}".format(self.batch_size));
        for layer in self.layers:
            layer.dump();
    ## main method of the network
    def _init_layers(self):
        # first input layer do not have weight
        for i in range(1,self.num_layers):
            self.layers[i].init_weight( self.layers[i-1].size );
            self.layers[i].id = i;
            assert self.layers[i].typ == "fully_connected"
    def __init__(self,layers,cost_func="cross_entropy",alpha=0.001
                ,solver="SGD",batch_size=10):
        self.layers = layers;
        self.num_layers = len(layers);
        self.batch_size = batch_size;
        self._init_layers();
        self.cost_func = cost_fun_factory(cost_func);
        self.solver = solver;
        self.alpha = alpha;
    def _train(self,x,y):
        x = np.asarray(x);
        accurate = np.asarray(y);
        predict = self.predict(x);
        cost = self.cost_func.get_value(predict,accurate);
        print ("accurate:{0}, predict:{1}, cost:{2}".format(accurate,predict,cost));
    def train(self,x,y):
        for _x,_y in zip(x,y):
            self._train(_x,_y);        
    def predict(self,x):
        for layer in self.layers:
            if layer.typ is "input":
                continue;
            temp_out = layer.eval(x);
        predict = temp_out;
        return predict;
    
