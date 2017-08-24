from layers.base_layers import Layer;

class FullyConnect(Layer):
    def __init__(self,size,activation="RELU"):
        self._init_para(size,activation);
    