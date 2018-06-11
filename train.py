import sys
import pickle
from collections import Counter

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CLEVR, collate_data, transform
from model import RelationNetworks

batch_size = 640
lr = 5e-6
lr_max = 5e-4
lr_gamma = 2
lr_step = 20
clip_norm = 50
reverse_question = True
weight_decay = 1e-4
n_epoch = 500
n_worker = 9
data_parallel = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epoch):
    train_set = DataLoader(
        CLEVR(
            sys.argv[1],
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
        ),
        batch_size=batch_size,
        num_workers=n_worker,
        shuffle=True,
        collate_fn=collate_data,
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    relnet.train(True)
    for i, (image, question, q_len, answer, _) in enumerate(pbar):
        image, question, q_len, answer = (
            image.to(device),
            question.to(device),
            torch.tensor(q_len),
            answer.to(device),
        )

        relnet.zero_grad()
        output = relnet(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        nn.utils.clip_grad_norm_(relnet.parameters(), clip_norm)
        optimizer.step()

        correct = output.data.cpu().numpy().argmax(1) == answer.data.cpu().numpy()
        correct = correct.sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}; LR: {:.6f}'.format(
                epoch + 1,
                loss.detach().item(),
                moving_loss,
                optimizer.param_groups[0]['lr'],
            )
        )


def valid(epoch):
    valid_set = DataLoader(
        CLEVR(
            sys.argv[1],
            'val',
            transform=None,
            reverse_question=reverse_question,
            use_preprocessed=True,
        ),
        batch_size=batch_size // 2,
        num_workers=4,
        collate_fn=collate_data,
    )
    dataset = iter(valid_set)

    relnet.eval()
    class_correct = Counter()
    class_total = Counter()

    with torch.no_grad():
        for image, question, q_len, answer, answer_class in tqdm(dataset):
            image, question, q_len = (
                image.to(device),
                question.to(device),
                torch.tensor(q_len),
            )

            output = relnet(image, question, q_len)
            correct = output.data.cpu().numpy().argmax(1) == answer.numpy()
            for c, class_ in zip(correct, answer_class):
                if c:
                    class_correct[class_] += 1
                class_total[class_] += 1

    class_correct['total'] = sum(class_correct.values())
    class_total['total'] = sum(class_total.values())

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(3)), 'w') as w:
        for k, v in class_total.items():
            w.write('{}: {:.5f}\n'.format(k, class_correct[k] / v))

    print('Avg Acc: {:.5f}'.format(class_correct['total'] / class_total['total']))


if __name__ == '__main__':
    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    relnet = RelationNetworks(n_words)
    if data_parallel:
        relnet = nn.DataParallel(relnet)
    relnet = relnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(relnet.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    for epoch in range(n_epoch):
        if scheduler.get_lr()[0] < lr_max:
            scheduler.step()

        train(epoch)
        valid(epoch)

        with open(
            'checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(3)), 'wb'
        ) as f:
            torch.save(relnet.state_dict(), f)
