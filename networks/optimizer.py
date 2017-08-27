import numpy as np;

class Optimizer:
    def _cal_layer_result(self,layers,x):
        # input,x: np matrix of one sample, 1 * n_features
        # mid,x: x.T -> n_features *1
        # output,layers_result: list of np array n_unit * 1
        layers_result = [];
        temp_out = x.T;
        for layer in layers:
            layers_result.append( layer.eval(temp_out) );            
        return layers_result;
    def _update_layers(self,layers,dw,db):
        # input,dw db: list of changes in W and b
        # output, stop_flag: whether the iteration should stop
        stop_flag = False;
        nlayers = len(layers);
        norm_w = 0;
        norm_b = 0;
        for i in range(1,nlayers):
            norm_w += np.linalg.norm(dw[i-1]);
            norm_b += np.linalg.norm(db[i-1]);
            # print(norm_w);
            layers[i].weight -= self.alpha * dw[i-1];
            layers[i].bias   -= self.alpha * db[i-1];
        if (norm_w < self.eps and norm_b < self.eps):
            stop_flag = True;
        return stop_flag;
    def _calculate_gradient(self,layers,layers_result,accurate):
        # input,accurate: np matrix, 1 * n_outputs
        # input,layers_result: list of np array n_units * 1
        # output,gradients: list a gradient n_units * 1
        nlayers = len(layers);
        gradients = (nlayers-1)*[0];
        # mid,gradient: 1 * n_outputs
        gradient = self.cost_func.get_gradient(layers_result[-1],accurate);
        for i in range(nlayers-1,0,-1):
            gradient = np.multiply(gradient, layers[i].unit.eval_deri(layers_result[i]));
            gradients[i-1] = gradient;
            gradient = layers[i].weight.T * gradient;
        return gradients;
    def _calculate_change(self,layers,layers_result,gradients):
        # input,gradients: list a gradient n_units * 1
        # input,layers_result: list of np array n_units * 1
        # output,dw db: list of changes in W and b
        nlayers = len(layers);
        dw = [None] * (nlayers-1);
        db = [None] * (nlayers-1);
        # skip input layer
        for i in range(1,nlayers):
            db[i-1] = gradients[i-1];
            dw[i-1] = db[i-1] * layers_result[i-1].T;
        return dw,db;
    def __init__(self,cost_func, alpha, eps):
        # record every cost in each iteration
        self.costs = [];
        self.cost_func = cost_func;
        self.alpha = alpha;
        self.eps = eps;
        # print(self.costs[-1]);
    def do_batch(self,layers,x,accurate):
        nlayers = len(layers);
        # input,x: np matrix, n_samples * n_features
        # input,accurate: np matrix, n_samples * n_outputs
        # output, finish_flag: whether the iteration should stop
        n_sample = x.shape[0];
        local_cost = 0.0;
        ## can be optimize via enabling multithread
        batch_dw = [None] * (nlayers-1);
        batch_db = [None] * (nlayers-1);
        for i in range(0,n_sample):
            # mid,layers_result: list of np array n_unit * 1
            layers_result = self._cal_layer_result(layers,x[i,:]);
            local_cost += self.cost_func.get_value(layers_result[-1],accurate[i,:]);
            gradients = self._calculate_gradient(layers,layers_result,accurate[i,:]);
            dw , db = self._calculate_change(layers,layers_result,gradients);
            for j in range(0,nlayers-1):
                if i is 0:
                    batch_dw[j]  = dw[j];
                    batch_db[j]  = db[j];                    
                else:
                    batch_dw[j] += dw[j];
                    batch_db[j] += db[j];
        local_cost /= n_sample;
        print(local_cost);
        for dw,db in zip(batch_dw,batch_db):
            dw/=n_sample;
            db/=n_sample;
        stop_flag = self._update_layers(layers,batch_dw,batch_db);
        return stop_flag,local_cost;