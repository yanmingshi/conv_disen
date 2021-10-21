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


class Wide2(nn.Module):

    def __init__(self, args):
        super(Wide2, self).__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.reg_weight = args.reg_weight
        self.BPRLoss = BPRLoss()
        self.L2Loss = L2Loss()

        self.feature_1 = nn.Conv2d(1, 1, kernel_size=(1, 1))
        self.feature_2 = nn.Conv2d(1, 1, kernel_size=(1, 2))
        self.feature_3 = nn.Conv2d(1, 1, kernel_size=(1, 4))
        self.feature_4 = nn.Conv2d(1, 1, kernel_size=(1, 8))
        self.feature_5 = nn.Conv2d(1, 1, kernel_size=(1, 16))

        self.linear = nn.Linear(5, 1)

    def get_features(self, embedding):
        f1 = torch.tanh(self.feature_1(embedding))
        f2 = torch.tanh(self.feature_2(embedding))
        f3 = torch.tanh(self.feature_3(embedding))
        f4 = torch.tanh(self.feature_4(embedding))
        f5 = torch.tanh(self.feature_5(embedding))
        return [f1, f2, f3, f4, f5]

    def calculate_score(self, u_features, i_features):

        scores = []
        u_batch = u_features[0].shape[0]
        i_batch = i_features[0].shape[0]
        for u, i in zip(u_features, i_features):
            u = u.view(u_batch, 1, -1)
            i = i.view(i_batch, -1, 1)
            scores.append(torch.matmul(u, i).squeeze())
        scores = torch.stack(scores, dim=1)
        scores = torch.sum(scores, dim=-1)
        return scores



    def forward(self, users, positives, negatives):
        users_embedding = self.user_embedding(users).view(-1, 1, 1, self.embedding_size)
        positives_embedding = self.item_embedding(positives).view(-1, 1, 1, self.embedding_size)
        negatives_embedding = self.item_embedding(negatives).view(-1, 1, 1, self.embedding_size)
        user_features = self.get_features(users_embedding)
        p_features = self.get_features(positives_embedding)
        n_features = self.get_features(negatives_embedding)

        p_score = self.calculate_score(user_features, p_features)
        n_score = self.calculate_score(user_features, n_features)

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
        u_features = self.get_features(users_embedding)
        i_features = self.get_features(items_embedding)

        score = []
        u_batch = u_features[0].shape[0]
        i_batch = i_features[0].shape[0]
        for u, i in zip(u_features, i_features):
            u = u.view(u_batch, 1, 1, -1)
            i = i.view(1, i_batch, -1, 1)
            score.append(torch.matmul(u, i).squeeze())
        score = torch.stack(score, dim=2)
        score = torch.sum(score, dim=-1)
        return score
