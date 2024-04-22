import sys
sys.path.append('.')
import argparse
import os
import pickle
from Datasets import SketchyDataset, TUBerlinDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
import pretrainedmodels
import torch.nn.functional as F
from ResnetModel import HashingModel
from logger import Logger
from utils import resume_from_checkpoint
from visualize import visualize_ranked_results
from itq import compressITQ
from utils import load_data, get_train_args



from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def evaluate(args, resume_dir, get_precision=False, model=None, recompute=False, visualize=False):
    args.resume_dir = resume_dir