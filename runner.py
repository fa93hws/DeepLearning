import math;

from networks.MLP import MLPRegressor;
from test.geometry import Parabola;
from layers.base_layers import Input, Output, FullyConnect;

def l2norm(x,y):
    dist = 0.0;
    for a,b in zip(x,y):
        dist += (a-b)**2;
    return math.sqrt(dist);
def geo_test():
    ## setup networks, 3 layers MLP
    layers = [];
    layers.append( Input(1) );
    layers.append( FullyConnect(1) );
    layers.append( Output(1,"linear") );
    regressor = MLPRegressor(layers);
    ## train network
    test = Parabola();
    train_data = test.getTrainData();
    regressor.train(train_data[0],train_data[1]);
    ## verify result
    # x = [-5,-9,-10,5,0.5,7,9];
    # y_predict = regressor.predict(x);
    # y_exact = test.getY(x);
    # dist = l2norm(y_predict, y_exact);
    # print(dist);



if __name__ == "__main__":
    geo_test();