'''
Plan:
1) Load dataset into data loaders √
2) Define model
3) Implement train and test functions
4) Run training
5) Save / load model
'''
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from dataset import FakeNewsDataset
from util import *
from baseline import NeuralNetwork

BATCH_SIZE=10

# Pre-processing
stances = pd.read_csv('train_stances.csv')
bodies = pd.read_csv('train_bodies.csv')
vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((stances['Headline'].values, bodies['articleBody'].values), axis=None))

# load dataset
training_data = FakeNewsDataset('train_stances.csv', 'train_bodies.csv')
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

# Model and loss
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

import wandb
# Set up wandb logging
wandb.init(
    # Set the project where this run will be logged
    project="test-logging", 
    # Track hyperparameters and run metadata
    config={
        "test string": "hello world",
        "value": 0.5
    }
)

# Training
size = len(train_dataloader.dataset)
model.train()
for batch, ((headline, body), y) in enumerate(train_dataloader):
    x_head = torch.Tensor(vectorizer.transform(headline).todense())
    x_body = torch.Tensor(vectorizer.transform(body).todense())

    # Compute prediction error
    pred = model(x_head, x_body)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * len(x_head)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        wandb.log({'loss': loss})