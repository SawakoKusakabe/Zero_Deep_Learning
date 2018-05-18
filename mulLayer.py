
# coding: utf-8

# In[ ]:


class MulLayer:
    def __int__(self):
        self.x = None
        self.y = None
        
        
    def forward (self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y # xとｙをひっくり返す
        dy = dout * self.x
        
        return dx, dy

