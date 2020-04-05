from matplotlib import pyplot as plt
import numpy as np
import torch as t
from argparse import Namespace

#set the random seed to make sure that we could have the same input
t.manual_seed(1000)

class LinearRegressionAutoGrad:
    def __init__(self):
        self.w = t.rand(1,1).requires_grad_()
        self.b = t.rand(1,1).requires_grad_()
        self.batch_size = 24
        self.lr = 0.005

        self.losses = []
        self.epochs = 10
    def generate_data(self):
        x = t.rand(self.batch_size,1) * 5
        y = 3 * x + 2 + t.rand(self.batch_size, 1)
        return (x,y)
    
    def training(self):

        for e in range(self.epochs):
                
            x,y = self.generate_data()

            #forward
            y_pred = x.mm(self.w) + self.b.expand_as(y)
            loss = 0.5 * (y_pred - y) ** 2
            print(y_pred - y)
            loss = loss.sum()
            self.losses.append(loss.item())

            print(loss)
            # backward
            loss.backward()

            # update the parameters
            self.w.data.sub_(self.lr * self.w.grad.data)
            self.b.data.sub_(self.lr * self.b.grad.data)

            # clear the gradient,others it will accumulate
            self.w.grad.data.zero_()
            self.b.grad.data.zero_()
        self.visualize()
    def visualize(self):
        x,y = self.generate_data()
        fake_x = t.arange(0,6).view(-1,1).float()
        y_pred = fake_x.mm(self.w.data) + self.b.data.expand_as(fake_x)
        plt.scatter(x.squeeze().numpy(),y.squeeze().numpy())
        plt.plot(fake_x.squeeze().numpy(),y_pred.numpy())
        plt.xlim(0,5)
        plt.ylim(0,16)
        plt.show()
    def visualize_loss(self):
        plt.plot(self.losses)
        plt.show()

lr = LinearRegressionAutoGrad()
lr.training()
        

