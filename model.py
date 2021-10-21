#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: yanms
# @Date  : 2021/8/25 22:06
# @Desc  :
import math

import torch
import torch.nn as nn

import torchvision


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score)).mean()
        return loss


class L2Loss(nn.Module):
    def __init__(self, norm=2):
        super(L2Loss, self).__init__()
        self.norm = norm

    def forward(self, model):
        loss = 0
        for params in model.parameters():
            loss += torch.norm(params, p=self.norm)
        return loss


class CNNNet(nn.Module):

    def __init__(self, args):
        super(CNNNet, self).__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.BPRLoss = BPRLoss()
        self.L2Loss = L2Loss()
        self.reg_weight = args.reg_weight
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(64, 64, kernel_size=(3, 3)),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1),

        )
        self.scores = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 1 * 1, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        if args.if_load_model:
            parameters = torch.load(args.model_full_name)
            self.load_state_dict(parameters)

    def forward(self, users, positives, negatives):
        users_embedding = self.user_embedding(users).view(-1, 1, 8, 8)
        positives_embedding = self.item_embedding(positives).view(-1, 1, 8, 8)
        negatives_embedding = self.item_embedding(negatives).view(-1, 1, 8, 8)
        positive_pair = torch.cat((users_embedding, positives_embedding), dim=1)
        negative_pair = torch.cat((users_embedding, negatives_embedding), dim=1)

        p_score = self.features(positive_pair)
        p_score = torch.flatten(p_score, 1)
        p_score = self.scores(p_score)

        n_score = self.features(negative_pair)
        n_score = torch.flatten(n_score, 1)
        n_score = self.scores(n_score)
        return p_score, n_score

    def calculate(self, batch_data):
        users = batch_data[:, 0]
        positives = batch_data[:, 1]
        negatives = batch_data[:, 2]
        p_score, n_score = self.forward(users, positives, negatives)
        loss = self.BPRLoss(p_score, n_score) + self.reg_weight * self.L2Loss(self)
        return loss

    def predict(self, users):

        user_embedding = self.user_embedding(users).view(-1, 1, 8, 8)
        items_embedding = self.item_embedding.weight.view(-1, 1, 8, 8)
        size = items_embedding.shape[0]
        # user_embedding = user_embedding.expand(-1, items_embedding.shape[0], -1, -1)
        # pair = torch.cat((user_embedding, items_embedding), dim=1)
        batch_size = 1024
        batch = math.ceil(size / batch_size)

        scores_list = []
        for i in range(users.shape[0]):
            single_scores = torch.zeros(size)
            user_embedding = self.user_embedding(users[i]).view(-1, 1, 8, 8).expand(size, -1, -1, -1)
            pair = torch.cat((user_embedding, items_embedding), dim=1)
            for j in range(batch):
                if batch_size * (j + 1) > size:
                    end = size

                else:
                    end = (j + 1) * batch_size
                scores = self.features(pair[j * batch_size: end])
                scores = torch.flatten(scores, 1)
                scores = self.scores(scores).squeeze()
                single_scores[j * batch_size: end] = scores
            scores_list.append(single_scores)
        return torch.stack(scores_list)
