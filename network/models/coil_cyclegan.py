from logger import coil_logger
import torch.nn as nn
import torch
import importlib

from configs import g_conf
from coilutils.general import command_number_to_index

from .building_blocks import Conv
from .building_blocks import Branching
from .building_blocks import FC
from .building_blocks import Join


class CoILCycleGAN(nn.Module):
    def __init__(self, params):
        super(CoILCycleGAN).__init__()
        self.params = params

    def forward(self, x):
        pass
