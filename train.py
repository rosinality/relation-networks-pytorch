import sys

import torch
from torch.utils.data import DataLoader

from dataset import CLEVR, collate_data, transform

train_set = DataLoader(CLEVR(sys.argv[1], transform=transform), batch_size=2, collate_fn=collate_data)

print(next(iter(train_set)))