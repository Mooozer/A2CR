# cv2==4.4.0
# gym==0.19.0
# scipy==1.1.0
# torch==1.13.1
# numpy==1.19.2
# pandas==1.2.1
# seaborn==0.11.2
# sklearn==0.24.2
# skimage==0.17.2
# IPython==7.19.0

import cv2
import gym
import copy
import time
import scipy 
import torch
import datetime
import gym_super_mario_bros
import random, os.path, math, glob, csv, base64, itertools, sys


import numpy as np 
import pandas as pd
import torch.nn as nn
import seaborn as sns
import torch.nn.functional as F 
import matplotlib.pyplot as plt


from math import gcd
from gym import spaces
from torch import optim
from pathlib import Path
from pprint import pprint
from collections import deque
from sklearn.utils import shuffle
from skimage.filters import window
from nes_py.wrappers import JoypadSpace
from IPython.display import clear_output
from IPython.core.debugger import set_trace
from torch.optim.lr_scheduler import StepLR
from IPython import display as ipythondisplay
from torch.distributions.categorical import Categorical
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
cv2.ocl.setUseOpenCL(False)