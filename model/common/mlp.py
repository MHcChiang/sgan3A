import torch.nn as nn
import torch
from utils.utils import initialize_weights

# class MLP(nn.Module):
    
#     def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
#         super().__init__()
#         if activation == 'tanh':
#             self.activation = torch.tanh
#         elif activation == 'relu':
#             self.activation = torch.relu
#         elif activation == 'sigmoid':
#             self.activation = torch.sigmoid

#         self.out_dim = hidden_dims[-1]
#         self.affine_layers = nn.ModuleList()
#         last_dim = input_dim
#         for nh in hidden_dims:
#             self.affine_layers.append(nn.Linear(last_dim, nh))
#             last_dim = nh

#         initialize_weights(self.affine_layers.modules())        

#     def forward(self, x):
#         for affine in self.affine_layers:
#             x = self.activation(affine(x))
#         return x


class MLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            act_layer = nn.Tanh()
        elif activation == 'relu':
            act_layer = nn.ReLU()
        elif activation == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU(negative_slope=0.01)
        else:
            act_layer = nn.Identity()

        self.out_dim = hidden_dims[-1]
        layers = []
        last_dim = input_dim
        for nh in hidden_dims:
            layers.append(nn.Linear(last_dim, nh))
            layers.append(act_layer) # 把 Activation 加入層列表中
            last_dim = nh
            
        self.model = nn.Sequential(*layers)

        initialize_weights(self.model.modules())        

    def forward(self, x):
        return self.model(x)