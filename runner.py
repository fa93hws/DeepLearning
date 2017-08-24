from networks.MLP import MLPRegressor;
from test.geometry import Parabola;
from layers.fully_connected import FullyConnect;
from layers.io import Input, Output;

def geo_test():
    ## setup networks, 3 layers MLP
    layers = [];
    layers.append( Input(1) );
    layers.append( FullyConnect(1) );
    layers.append( Output(1) );

    ## verify result
    test = Parabola();
    train_data = test.getTrainData();

if __name__ == "__main__":
    geo_test();