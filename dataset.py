import os
import json

from PIL import Image
from torch.utils import data

class CLEVR(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(os.path.join(root, questions,
                            f'CLEVR_{mode}_questions.json')) as f:
            self.data = json.load(f)['questions']

        self.transform = transform