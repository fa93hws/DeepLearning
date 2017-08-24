class Parabola:
    # x,y
    def _fun(self,x):
        return 9*x**2 + x*4 - 12;
    def __init__(self):
        self._init();
    def _init(self):
        self.x = [1,10,100,2,20,9,45,-2,-10,2,945,2,0.2,1.5];
        self.y = [self._fun(a) for a in self.x];
    def getTrainData(self):
        return [self.x,self.y];
    def getY(self,x):
        return [self._fun(a) for a in x];
    def getPara(self):
        return [9,1,-12];
