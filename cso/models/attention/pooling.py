import torch.nn as nn


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        pass
            
    def forward(self, tokens):
        x = tokens.mean(dim=1)
        return x