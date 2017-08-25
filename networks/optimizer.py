
class Optimizer:
    def _cal_layer_result(self,layers,x):
        # x: np tensor store one training smaple
        # return all layer results for one sample, list of matrix
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
    def _update_layers(self,layers,y,all_results,solver):
        
    def _optimize_once(self,layers,x,y,solver):
        # input,x: a list of np 2d array,based on row (if input is a list)
        # input,y: a list of np row array (if input is a list)
        all_results = self._cal_sample_result(layers,x,solver);
        n_layers = len(layers);
        all_predict = [results[n_layers-1] for results in all_results];
        local_cost = solver.cost_func.get_value(all_predict,y);
        self.costs.append(local_cost);
        # print(local_cost);
    def __init__(self,layers,x,y,solver):
        # record every cost in each iteration
        self.costs = [];

        self._optimize_once(layers,x,y,solver);