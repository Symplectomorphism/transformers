import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

import math
import random
from create_dataset import create_dataset

class StateActionReturnDataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 10 * 3
        return C

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)   # (block_size, state_dim=1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)   # (block_size, action_dim=1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)   # (block_size, 1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1) 

        return states, actions, rtgs, timesteps

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(10*1000)

train_dataset = StateActionReturnDataset(obss, 10 * 3, actions, done_idxs, rtgs, timesteps)

# Set up configs
C = CN()

## system
C.system = CN()
C.system.seed = 3407
C.system.work_dir = './out/decgpt'

# data
C.data = StateActionReturnDataset.get_default_config()

# model
C.model = GPT.get_default_config()
C.model.model_type = 'gpt-mini'
C.model.vocab_size = train_dataset.vocab_size
C.model.block_size = train_dataset.block_size
C.model.max_timestep = max(timesteps)

# trainer
C.trainer = Trainer.get_default_config()
C.trainer.learning_rate = 5e-4
C.trainer.max_epochs = 50
C.trainer.num_workers = 4

model = GPT(C.model)

# initialize a trainer instance and kick off training
trainer = Trainer(model, train_dataset, C.trainer)
trainer.run()