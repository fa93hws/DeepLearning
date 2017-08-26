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
        return temp.out;
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
    # def _update_layers(self,layers,chain):
    #     nlayers = len(layers);
    #     finish_flag = False;
    #     for i in range(nlayers-1,1,-1):
    #         dw = chain["dw"][i];
    #         db = chain["db"][i];
    #         layers[i].weight += self.alpha * dw;
    #         layers[i].bias   += self.alpha * db;
    #         if not finish_flag:
    #             norm_dw = np.linalg.norm(dw);
    #             norm_db = np.linalg.norm(db);
    #             # print(norm_dw);
    #             if norm_dw < self.eps and norm_db < self.eps:
    #                 finish_flag = True;
    #     return finish_flag;
    def __init__(self,cost_func, alpha, eps):
        # record every cost in each iteration
        self.costs = [];
        self.cost_func = cost_func;
        self.alpha = alpha;
        self.eps = eps;
        # print(self.costs[-1]);
    def do_once(self,layers,x,y):
        nlayers = len(layers);
        # input,x: np matrix, n_samples * n_features
        # input,y: np matrix, n_samples * n_outputs
        # output, finish_flag: whether the iteration should stop
        all_results = self._cal_sample_result(layers,x);
        # all_predict = [results[nlayers-1] for results in all_results];
        # local_cost = self.cost_func.get_value(all_predict,y);
        # self.costs.append(local_cost);
        # # print(local_cost);
        # chain = self.cost_func.get_gradient(layers,x,y,all_results);
        # finish_flag = self._update_layers(layers,chain);
        return finish_flag
        