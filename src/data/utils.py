import torch
import torchvision
import pandas as pd
import numpy as np
import sys

def print_versions():
    print('System Version:', sys.version)
    print('PyTorch version:', torch.__version__)
    print('Torchvision version:', torchvision.__version__)
    print('Numpy version:', np.__version__)
    print('Pandas version:', pd.__version__)
