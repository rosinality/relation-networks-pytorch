import sys
import pickle

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVR, collate_data, transform
from model import RelationNetworks

batch_size = 64
n_epoch = 180

with open('data/dic.pkl', 'rb') as f:
    dic = pickle.load(f)

n_words = len(dic['word_dic']) + 1
n_answers = len(dic['answer_dic'])

relnet = RelationNetworks(n_words).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(relnet.parameters(), lr=2.5e-4)

train_set = DataLoader(CLEVR(sys.argv[1], transform=transform),
                    batch_size=batch_size, num_workers=4,
                    collate_fn=collate_data)

for epoch in range(n_epoch):
    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    for image, question, q_len, answer in pbar:
        image, question, answer = \
            Variable(image).cuda(), Variable(question).cuda(), \
            Variable(answer).cuda()

        relnet.zero_grad()
        output = relnet(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        
        if moving_loss == 0:
            moving_loss = loss.data[0]

        else:
            moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1
        
        pbar.set_description('Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'. \
                            format(epoch + 1, loss.data[0], moving_loss))
