import numpy as np;

class Optimizer:
    def _cal_layer_result(self,layers,x):
        # input,x: np matrix of one sample, 1 * n_features
        # mid,x: x.T -> n_features *1
        # output,y: np array of one sample, n_outputs * 1
        for layer in layers:
            if layer.typ is "input":
                # tranpose the input in the input layer
                temp_out = x.T;
                continue;
            temp_out = layer.eval(temp_out);
        return temp_out;
    def _cal_sample_result(self,layers,x):
        # input ,x: np matrix, n_samples * n_features
        # output,y: np matrix, n_samples * n_outputs
        n_samples = x.shape[0];
        first_result = self._cal_layer_result(layers,x[0,:]);
        n_outputs = first_result.shape[1];
        y = np.asmatrix( np.zeros((n_samples,n_outputs)) );
        y[0,:] = first_result;
        ## can be optimize via enabling multithread
        for i in range(1,n_samples):
            y[i,:] = self._cal_layer_result(layers,x[i,:]);
        return y;
    def _update_layers(self,layers,x,predict,accurate):
        # input,x: np matrix, n_samples * n_features
        # input,accurate: np matrix, n_samples * n_outputs
        # input,predicts: np matrix, n_samples * n_outputs
        # output, finish_flag: bool, whether the iteration should stop
        finish_flag = false;
        # gradient dJ/dy(hat) at output layer
        gradient = self.cost_func.get_gradient(predict,accurate);
        gradient = np.multiply(layers[-1].unit.eval_deri())
        return finish_flag;
    def __init__(self,cost_func, alpha, eps):
        # record every cost in each iteration
        self.costs = [];
        self.cost_func = cost_func;
        self.alpha = alpha;
        self.eps = eps;
        # print(self.costs[-1]);
    def do_once(self,layers,x,accurate):
        nlayers = len(layers);
        # input,x: np matrix, n_samples * n_features
        # input,accurate: np matrix, n_samples * n_outputs
        # output, finish_flag: whether the iteration should stop
        # mid,predicts: np matrix, n_samples * n_outputs
        predicts = self._cal_sample_result(layers,x);
        local_cost = self.cost_func.get_value(predicts,accurate);
        self.costs.append(local_cost);
        finish_flag = self._update_layers(layers,x,predict,accurate);
        return finish_flag
        