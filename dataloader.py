from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import csv

class CellDataset(Dataset):
    def __init__(self, root, mode, transforms=None):

