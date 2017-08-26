import math;
from abc import ABCMeta,abstractmethod;
import numpy as np;

def cost_fun_factory(options):
    # default value of typ is cross_entropy
    if options.get("typ","cross_entropy") is "cross_entropy":
        return CrossEntropy(options);
    elif options["typ"] is "l2norm":
        return L2Dist(options);
    else:
        raise Exception("Cost function type not found",
                        "There is no cost function type name {0}".format(options.typ));

class CostFun(metaclass=ABCMeta):
    @abstractmethod
    def dump(self):
        print("Type of cost function is {0}".format(self.typ));
        print("Weight decay factor is {0}".format(self.decay));
    @abstractmethod
    def get_value(self,predict,accurate):
        # predict: a list of np row array
        # accurate: a list of np row array
        pass;
    @abstractmethod
    def get_gradient(self,layers,x,predict,accurate):
        # input is numpy array
        pass;
    def __init__(self,options):
        self.decay = options.get("lambda",0);
        self.typ = options.get("typ","cross_entropy");


class CrossEntropy(CostFun):
    def dump(self):
        super(CrossEntropy, self).dump();
    def __init__(self,options):
        super(CrossEntropy, self).__init__(options);
    def get_value(self,predict,accurate):
        pass;
    def get_gradient(self,layers,x,predict,accurate):
        pass;

class L2Dist(CostFun):
    def dump(self):
        super(CrossEntropy, self).dump();
    def __init__(self,options):
        super(L2Dist, self).__init__(options);
    def get_value(self,predict,accurate):
        # predict: a list of np row array
        # accurate: a list of np row array
        cost = 0.0;
        len = 0;
        for p,a in zip(predict,accurate):
            len += 1;
            cost += np.linalg.norm(p-a);
        return cost/len/2;
    # def _get_layer_sample_gradient(self,w,b,x,m,n,accurate,leading_term):
    #     ## can be optimized via using multi-threading
    #     # m-th column
    #     dw = np.zeros((n,m));
    #     for i in range(0,m):
    #         dw[:,i] = np.multiply(leading_term,x.T);
    #     db = leading_term;
    #     return dw,db;
    # def _get_layer_gradient(self,layer,x,accurate):
    #     #### Input
    #     # W*x_i + b => y_i
    #     # accurate, expected y, list of m*1 np array
    #     # x: list of n*1 np array
    #     #### Output
    #     # dw: change in weight,np matrix, n*m
    #     # db: change in bias, n*1 np array
    #     n = x[0].shape[0];
    #     m = accurate[0].shape[0];
    #     num_sample = len(x);
    #     dw = np.zeros((n,m));
    #     db = np.zeros((n,1));
    #     ## can be optimized via using multi-threading
    #     for a,y in zip(x,accurate):
    #         leading_term = y - (layer.weight * a + layer.bias);
    #         dw_iter, db_iter = self._get_layer_sample_gradient(
    #                             layer.weight,layer.bias,a,m,n,y,leading_term);
    #         dw += dw_iter;
    #         db += db_iter;
    #     dw /= num_sample;
    #     db /= num_sample;
    #     return dw,db;
    # def get_gradient(self,layers,x,accurate,all_results):
    #     n_layers = len(layers);
    #     # all results in i-th layer with all sample
    #     # a list of a list of n*1 np array
    #     layer_results = n_layers*[None];
    #     for i in range(0,n_layers):
    #         layer_results[i] = [results[i] for results in all_results];        
    #     # store gradient for every layer in chain
    #     # first value in chain -> gradient in last layer
    #     chain = {"dw":n_layers*[None],"db":n_layers*[None]};
    #     # output layer
    #     y = accurate;
    #     dw,db = self._get_layer_gradient( layers[n_layers-1], layer_results[n_layers-1], y);
    #     chain["dw"][n_layers-1] = dw;
    #     chain["db"][n_layers-1] = db;
    #     # for i in range(n_layers-1,0):
    #     #     print(i)
    #     return chain;
    def get_gradient(self,predict,accurate):

        m = predict.shape[0];
        return predict-accurate;
