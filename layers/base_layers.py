from layers.units import unit_factory;

class Layer:
    def _init_units(self,size,typ):
        self.units = [];
        for i in range (1,size):
            self.units.append(unit_factory(typ));
    def _init_weight(self,size,typ):
        self.weight = [];
        for i in range (1,size):
            self.weight.append(0.1);
    def _init_para(self,size,typ):
        self.size = size;
        self.typ = typ;
        self._init_units(size,self.typ);
        self._init_weight(size,self.typ);

class Input(Layer):
    def __init__(self,size):
        self._init_para(size,"linear");


class Output(Layer):
    def __init__(self,size,typ):
        self._init_para(size,typ);

