"""
This is mostly a copy of https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

GPT model:
- Starts with token and positional embedding
- Transformer blocks
  - Sequential combination of a 1-hidden-layer MLP (KAN?) and a self-attention block
  - All blocks feed into a central residual pathway similar to ResNets
- The final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from mingpt.utils import CfgNode as CN