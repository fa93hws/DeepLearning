
class MLPRegressor:
    def __init__(self,layers,cost_func="cross_entropy",alpha=0.001
                ,solver="SGD",batch_size=10):
        self.layers = layers;
        self.cost_func = cost_func;
        self.solver = solver;
        self.alpha = alpha;
    def train(self,x,y):
        print("to be completed");
    def predict(self,x):
        print("to be completed");