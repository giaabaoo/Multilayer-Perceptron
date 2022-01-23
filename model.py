import torch
import torch.nn as nn  
import torch.nn.functional as F     

class MLP(nn.Module):
    def __init__(self, n_input, n_units, loss):
        super().__init__()
        self.layer1 = nn.Linear(n_input, n_units)
        if loss == "CE":
            self.layer2 = nn.Linear(n_units, 2)
        elif loss == "L2":
            self.layer2 = nn.Linear(n_units, 1)
        self.activation = nn.ReLU()

    def forward(self, input_x):
        output = self.layer1(input_x)
        output = self.activation(output)
        output = self.layer2(output)
        return output

class MultiMLP(nn.Module):
    def __init__(self, n_input, drop_out_rate):
        super(MultiMLP, self).__init__()
        self.layer1 = nn.Linear(n_input, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer_output = nn.Linear(32, 2)
        self.drop_out_rate = drop_out_rate

        if self.drop_out_rate:
            self.layer_dropout = nn.Dropout(p=self.drop_out_rate)
        
        self.activation = nn.ReLU()

    def forward(self, input_x):
        if self.drop_out_rate:
            output = self.layer_dropout(self.layer1(input_x))
            output = self.activation(output)
            output = self.layer_dropout(self.layer2(output))
            output = self.activation(output)
            output = self.layer_dropout(self.layer3(output))
            output = self.activation(output)
            output = self.layer_output(output)

        else:
            output = self.layer1(input_x)
            output = self.activation(output)
            output = self.layer2(output)
            output = self.activation(output)
            output = self.layer3(output)
            output = self.activation(output)
            output = self.layer_output(output)

        return output
