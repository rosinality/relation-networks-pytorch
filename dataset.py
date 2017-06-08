import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transforms import Scale

class CLEVR(Dataset):
    def __init__(self, root, split='train', transform=None):
        with open(f'data/{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)

        self.data
        self.transform = transform
        self.root = root
        self.split = split

    def __getitem__(self, index):
        imgfile, question, answer = self.data[index]
        img = Image.open(os.path.join(self.root, 'images', self.split, imgfile))

        img = self.transform(img)

        return img, question, answer

    def __len__(self):
        return len(self.data)

transform = transforms.Compose([
    Scale([128, 128]),
    transforms.Pad(4),
    transforms.RandomCrop([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

def collate_data(batch):
    images, answers = [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, answer = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        answers.append(answer)

    return torch.stack(images), torch.from_numpy(questions), \
        torch.LongTensor(answers)
