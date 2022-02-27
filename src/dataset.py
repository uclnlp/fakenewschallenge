# See: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import torch
from torch.utils.data import Dataset
import pandas as pd

STANCE_MAP = {
    'agree':0,
    'disagree':1,
    'discuss':2,
    'unrelated':3,
}

class FakeNewsDataset(Dataset):
    def __init__(self, stances_file, bodies_file):
        self.stances = pd.read_csv(stances_file)
        self.bodies = pd.read_csv(bodies_file)

    def __len__(self):
        return len(self.stances)
    
    def __getitem__(self, idx):
        headline, body_id, stance = self.stances.iloc[idx]
        select = self.bodies['Body ID'] == body_id
        body = self.bodies[select]['articleBody'].values[0]
        return (headline, body), STANCE_MAP[stance]

if __name__ == "__main__":
    training_data = FakeNewsDataset('train_stances.csv', 'train_bodies.csv')
    training_data.__getitem__(4)