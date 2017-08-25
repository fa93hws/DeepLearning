import random;

class Parabola:
    # x,y
    def _fun(self,x):
        return [[self.coe2 * (a[0]**2) + self.coe1 * a[0] - self.coe0] for a in x];
    def __init__(self):
        self.coe0 = (random.random()-0.5) * 10;
        self.coe1 = (random.random()-0.5) * 10;
        self.coe2 = (random.random()-0.5) * 10;
        self.x = [];
        for i in range (0,1000):
            self.x.append([ (random.random()-0.5)  * 100 ]);
        self.y = self._fun(self.x);
    def getTrainData(self):
        return [self.x,self.y];
    def getY(self,x):
        return self._fun(x);
    def getPara(self):
        return [self.coe2, self.coe1, self.coe0];
