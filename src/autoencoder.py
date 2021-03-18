import numpy as np
import torch
import torch.autograd as ag
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from scipy.optimize import minimize, NonlinearConstraint
import random

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from math import sqrt
import time

