import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.log_soft = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        return self.log_soft(self.linear(x))