from matplotlib import pyplot as plt
import numpy as np
import torch as t
from argparse import Namespace

class LinearRegression:
    def __init__(self):
        self.w = t.rand(1,1)
        self.b = t.rand(1,1)
        self.lr = 0.02
        self.batch_size = 24

        self.loss = 0
        self.epochs = 20
    
    def generate_data(self,batch_size = 4):
        x = t.rand(batch_size,1) * 5
        y = x *2 + 3 + t.rand(batch_size,1)
        return (x,y)
    
    def visualize(self):
        x,y = self.generate_data(batch_size = 30)
        fake_x = t.arange(0,6).view(-1,1).float()
        y_pred = fake_x.mm(self.w) + self.b.expand_as(fake_x)
        plt.scatter(x.squeeze().numpy(),y.squeeze().numpy())
        plt.plot(fake_x.squeeze().numpy(),y_pred.numpy())
        plt.xlim(0,5)
        plt.ylim(0,16)
        plt.show()

    def training(self):
        for _ in range(self.epochs):
            x,y = self.generate_data(batch_size = 24)
            # forward
            y_pred = x.mm(self.w) + self.b.expand_as(y)
            # compute the loss
            delta = (y - y_pred)
            loss = 0.5 * delta ** 2
            self.loss = loss.mean().item()

            #backward
            delta /= 24.0
            w_grad = -1 * x.t().mm(delta).expand_as(self.w)
            b_grad = -1 * delta.sum().expand_as(self.b)

            #update the parameter
            print(w_grad,self.w)
            self.w.sub_(self.lr*w_grad)
            self.b.sub_(self.lr*b_grad)

        self.visualize()

lr = LinearRegression()
lr.training()