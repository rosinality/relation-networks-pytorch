import os
import pickle

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


resize = transforms.Resize([128, 128])

transform = transforms.Compose([
    transforms.Pad(8),
    transforms.RandomCrop([128, 128]),
    transforms.RandomRotation(2.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

category = {'0': 'count',
            '1': 'count',
            '2': 'count',
            '3': 'count',
            '4': 'count',
            '5': 'count',
            '6': 'count',
            '7': 'count',
            '8': 'count',
            '9': 'count',
            '10': 'count',
            'blue': 'color',
            'brown': 'color',
            'cyan': 'color',
            'yellow': 'color',
            'gray': 'color',
            'green': 'color',
            'purple': 'color',
            'red': 'color',
            'rubber': 'material',
            'metal': 'material',
            'large': 'size',
            'small': 'size',
            'cylinder': 'shape',
            'cube': 'shape',
            'sphere': 'shape',
            'no': 'exist',
            'yes': 'exist'}


class CLEVR(Dataset):
    def __init__(self, root, split='train', transform=None,
                 reverse_question=False, use_preprocessed=False):
        with open(f'data/{split}.pkl', 'rb') as f:
            self.data = pickle.load(f)

        with open('data/dic.pkl', 'rb') as f:
            self.dic = pickle.load(f)
        self.answer_class = {v: k for k, v in self.dic['answer_dic'].items()}

        self.transform = transform
        self.root = root
        self.split = split
        self.reverse_question = reverse_question
        self.use_preprocessed = use_preprocessed

    def __getitem__(self, index):
        imgfile, question, answer, _ = self.data[index]

        if self.use_preprocessed is False:
            img = Image.open(os.path.join(self.root, 'images',
                                          self.split, imgfile)).convert('RGB')
            img = resize(img)

        else:
            img = Image.open(os.path.join(self.root, 'images',
                                          self.split + '_preprocessed',
                                          imgfile)).convert('RGB')

        answer_class = category[self.answer_class[answer]]

        if self.transform is not None:
            img = self.transform(img)

        else:
            img = eval_transform(img)

        if self.reverse_question:
            question = question[::-1]

        return img, question, len(question), answer, answer_class

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    images, lengths, answers, answer_classes = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer, class_ = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        answer_classes.append(class_)

    return torch.stack(images), torch.from_numpy(questions), \
        lengths, torch.LongTensor(answers), answer_classes
