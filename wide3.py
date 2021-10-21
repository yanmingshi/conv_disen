#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: yanms
# @Date  : 2021/8/25 22:06
# @Desc  :

import torch
import torch.nn as nn


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


class Wide3(nn.Module):

    def __init__(self, args):
        super(Wide3, self).__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.reg_weight = args.reg_weight
        self.BPRLoss = BPRLoss()
        self.L2Loss = L2Loss()

        self.feature = nn.Conv2d(1, 5, kernel_size=(1, 4))

        self.linear = nn.Linear(5, 1)

    def get_features(self, embedding):
        return torch.relu(self.feature(embedding))


    def forward(self, users, positives, negatives):
        users_embedding = self.user_embedding(users).view(-1, 1, 1, self.embedding_size)
        positives_embedding = self.item_embedding(positives).view(-1, 1, 1, self.embedding_size)
        negatives_embedding = self.item_embedding(negatives).view(-1, 1, 1, self.embedding_size)
        user_features = self.get_features(users_embedding)
        p_features = self.get_features(positives_embedding).transpose(2, 3)
        n_features = self.get_features(negatives_embedding).transpose(2, 3)

        p_score = torch.matmul(user_features, p_features).squeeze()
        p_score = self.linear(p_score).squeeze()
        n_score = torch.matmul(user_features, n_features).squeeze()
        n_score = self.linear(n_score).squeeze()

        return p_score, n_score

    def calculate(self, batch_data):
        users = batch_data[:, 0]
        positives = batch_data[:, 1]
        negatives = batch_data[:, 2]
        p_score, n_score = self.forward(users, positives, negatives)
        loss = self.BPRLoss(p_score, n_score) + self.reg_weight * self.L2Loss(self)
        return loss

    def predict(self, users):
        users_embedding = self.user_embedding(users).view(-1, 1, 1, self.embedding_size)
        items_embedding = self.item_embedding.weight.view(-1, 1, 1, self.embedding_size)
        u_features = self.get_features(users_embedding).unsqueeze(1)
        i_features = self.get_features(items_embedding).unsqueeze(0).transpose(-1, -2)
        score = torch.matmul(u_features, i_features).squeeze()
        score = self.linear(score).squeeze()
        return score
