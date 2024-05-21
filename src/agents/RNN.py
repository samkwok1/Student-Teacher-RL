import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZES = []
class MazeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=4, num_layers=3) -> None:
        super(MazeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # Three-layered RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        



