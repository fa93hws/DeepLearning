from networks.cost_func import cost_fun_factory;

class MLPRegressor:
    ## print details of the network
    def dump(self):
        print("It is a MLP regressor");
        print("It has %d layers", self.num_layers);
        print("solver type is %s", self.solver);
        print("Learning rate is %f", self.alpha);
        print("Cost function is %s", self.cost_func);
        print("batch_size is %d", self.batch_size);
        for layer in self.layers:
            print_layer(layer.dump());
    ## main method of the network
    def _init_layers(self):
        # first input layer do not have weight
        for i in range(2,self.num_layers):
            self.layers[i].init_weight( self.layers[i-1].size );
    def __init__(self,layers,cost_func="cross_entropy",alpha=0.001
                ,solver="SGD",batch_size=10):
        self.layers = layers;
        self.num_layers = len(layers);
        self.batch_size = batch_size;
        self._init_layers();
        self.cost_func = cost_fun_factory(cost_func);
        self.solver = solver;
        self.alpha = alpha;
    def train(self,x,y):
        print("to be completed");
    def predict(self,x):
        print("to be completed");
    
