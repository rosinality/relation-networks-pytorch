import os
import pickle

from PIL import Image
from torch.utils import data

class CLEVR(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(f'data/{mode}.pkl', 'wb') as f:
            self.data = pickle.load(f)

        self.transform = transform