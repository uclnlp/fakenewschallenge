import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(46673, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x_head, x_body):
        similarity = F.cosine_similarity(x_head, x_body)
        similarity = torch.unsqueeze(similarity, dim=-1)
        x = torch.cat((x_head, similarity, x_body), 1)
        logits = self.linear_relu_stack(x)
        return logits