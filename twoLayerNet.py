
# coding: utf-8

# In[12]:


import numpy as np
from rayers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_saize, hidden_size, output_size, weight_std=0.01):
        #　重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        
        # レイヤーの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['B1'])
        self.layers['Relu1']
        self.layers['arrine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer,forward(x)
            return x
        
    # ｘ：入力データ　ｔ：教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    #入力データ、教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(lossW, self.params['W1'])
        grads['b1'] = numerical_gradient(lossW, self.params['b1'])
        grads['W2'] = numerical_gradient(lossW, self.params['W2'])
        grads['b2'] = numerical_gradient(lossW, self.params['b2'])
        
        return grads
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        gads['W1'] = self.layers['Affine1'].dW
        gads['b1'] = self.layers['Affine1'].db
        gads['W2'] = self.layers['Affine2'].dW
        gads['b2'] = self.layers['Affine2'].db
                                     
        return grads

