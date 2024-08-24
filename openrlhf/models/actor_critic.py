from typing import Optional, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F

from openrlhf.models.actor import Actor


class ActorWithCritic(Actor):
    def __init__(
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

    