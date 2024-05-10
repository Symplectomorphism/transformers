"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import time
import logging
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        C.max_epochs = 10
        # device to train on
        C.device = 'auto'
        # dataloader parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        C.lr_decay = False
        C.warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
        C.final_tokens = 260e9 # (at what point we reach 10% of original LR)
        return C
    
    def __init__(self, model, train_dataset, config):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # # variables that will be assigned to trainer class later for logging and etc.
        # self.iter_num = 0
        # self.iter_time = 0.0
        # self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callback[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callback[onevent] = [callback]
    
    def trigger_callback(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        def run_epoch(epoch_num=0):
            # setup the dataloader
            train_loader = DataLoader(
                self.train_dataset,
                # sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            model.train()
            # self.iter_num = 0
            # self.iter_time = time.time()
            # data_iter = iter(train_loader)

            losses = []
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for it, (x, y, r, t) in pbar:
                # place the data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                # backprop and update the parameters
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
        # best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):
            run_epoch(epoch_num=epoch)