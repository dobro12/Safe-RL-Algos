import torch.nn as nn
import numpy as np
import torch

def initWeights(m, init_bias=0.0):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_bias, 0.01)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, shape, activation):
        super(MLP, self).__init__()
        self.activation_fn = activation
        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, input):
        return self.architecture(input)
