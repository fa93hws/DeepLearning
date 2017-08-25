import numpy as np;

class Optimizer:
    def _cal_layer_result(self,layers,x):
        # x: np tensor store one training smaple
        # return all layer results for one sample, list of n*1 array
        sample_results = [];
        for layer in layers:
            if layer.typ is "input":
                # tranpose the input in the input layer
                temp_out = x.T;
                sample_results.append(x);
                continue;
            temp_out = layer.eval(temp_out);
            sample_results.append(temp_out);
        return sample_results;
    def _cal_sample_result(self,layers,x,solver):
        # x: list of np tensor
        # return all sample results, list of list of matrix
        all_results = [];
        ## can be optimize via enabling multithread
        for a in x:
            all_results.append( self._cal_layer_result(layers,a) );
        return all_results;
    def _update_layers(self,layers,chain,solver):
        finish_flag = False;
        for i in range(self.nlayers-1,1,-1):
            dw = chain["dw"][i];
            db = chain["db"][i];
            layers[i].weight += solver.alpha * dw;
            layers[i].bias   += solver.alpha * db;
            if not finish_flag:
                norm_dw = np.linalg.norm(dw);
                norm_db = np.linalg.norm(db);
                print(norm_dw);
                if norm_dw < solver.eps and norm_db < solver.eps:
                    finish_flag = True;
        return finish_flag;
    def _optimize_once(self,layers,x,y,solver):
        # input,x: a list of np 2d array,based on row (if input is a list)
        # input,y: a list of np row array (if input is a list)
        all_results = self._cal_sample_result(layers,x,solver);
        all_predict = [results[self.nlayers-1] for results in all_results];
        local_cost = solver.cost_func.get_value(all_predict,y);
        self.costs.append(local_cost);
        # print(local_cost);
        chain = solver.cost_func.get_gradient(layers,x,y,all_results);
        finish_flag = self._update_layers(layers,chain,solver);
        return finish_flag
    def __init__(self,layers,x,y,solver):
        # record every cost in each iteration
        self.costs = [];
        self.nlayers = len(layers);
        for i in range(1, solver.max_iter):
            finish_flag = self._optimize_once(layers,x,y,solver);
            if finish_flag:
                break;
        print(self.costs[-1]);