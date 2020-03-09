import torch.nn as nn
import torch as t


class SurnameClassifier(nn.Module):
    def __init__(self, in_size,hidden_size,out_size = 2, ):
        """Define the structure of the network, the number of layers, the type of layers
        
        Arguments:
            nn {torch,nn} -- [inherit nn.Module to self-define the layer]
            in_size {[int]} -- [feature dimension]
            hidden_size {[int]} -- [the number of hidden units]
        
        Keyword Arguments:
            out_size {int} -- [description] (default: {2})
        """

        super(SurnameClassifier,self).__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(in_size, hidden_size))
        self.module_list.append(nn.Linear(hidden_size, out_size))

    def forward(self,x_in,apply_softmax = False):
        for module in self.module_list:
            x_in = module(x_in)
        if apply_softmax:
            # dim = 1 means applying softmax to each row
            softmax = nn.Softmax(dim = 0)
            x_in = softmax(x_in)
        return x_in

# data = t.rand(2,2)
# test = SurnameClassifier(in_size = 2,hidden_size = 2)
# out = test(data,apply_softmax = True)
# print(out)
# print(out.size())
# out2 = test(data)
# print(out2)