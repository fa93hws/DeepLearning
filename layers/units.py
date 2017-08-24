class RELU:
    def __init__(self,val):
        self.type="RELU";
        self.val = val;
    def get(self):
        return max(0,self.val);