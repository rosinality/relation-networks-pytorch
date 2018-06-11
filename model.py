import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, normal_
import torch.nn.functional as F


class RelationNetworks(nn.Module):
    def __init__(
        self,
        n_vocab,
        conv_hidden=24,
        embed_hidden=32,
        lstm_hidden=128,
        mlp_hidden=256,
        classes=29,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
        )

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.n_concat = conv_hidden * 2 + lstm_hidden + 2 * 2

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, classes),
        )

        self.conv_hidden = conv_hidden
        self.lstm_hidden = lstm_hidden
        self.mlp_hidden = mlp_hidden

        coords = torch.linspace(-4, 4, 8)
        x = coords.unsqueeze(0).repeat(8, 1)
        y = coords.unsqueeze(1).repeat(1, 8)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(self, image, question, question_len):
        conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        _, (h, c) = self.lstm(embed_pack)
        h_tile = h.permute(1, 0, 2).expand(
            batch_size, n_pair * n_pair, self.lstm_hidden
        )

        conv = torch.cat([conv, self.coords.expand(batch_size, 2, conv_h, conv_w)], 1)
        n_channel += 2
        conv_tr = conv.view(batch_size, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(batch_size, n_pair, n_pair, n_channel)
        conv2 = conv_tr.unsqueeze(2).expand(batch_size, n_pair, n_pair, n_channel)
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)

        concat_vec = torch.cat([conv1, conv2, h_tile], 2).view(-1, self.n_concat)
        g = self.g(concat_vec)
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden).sum(1).squeeze()

        f = self.f(g)

        return f
