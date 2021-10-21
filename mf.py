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


class MF(nn.Module):

    def __init__(self, args):
        super(MF, self).__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.BPRLoss = BPRLoss()
        self.L2Loss = L2Loss()
        self.reg_weight = args.reg_weight

        if args.if_load_model:
            parameters = torch.load(args.model_full_name)
            self.load_state_dict(parameters)

    def forward(self, users, positives, negatives):
        users_embedding = self.user_embedding(users).view(-1, 1, self.embedding_size)
        positives_embedding = self.item_embedding(positives).view(-1, self.embedding_size, 1)
        negatives_embedding = self.item_embedding(negatives).view(-1, self.embedding_size, 1)

        p_score = torch.matmul(users_embedding, positives_embedding).squeeze()

        n_score = torch.matmul(users_embedding, negatives_embedding).squeeze()
        return p_score, n_score

    def calculate(self, batch_data):
        users = batch_data[:, 0]
        positives = batch_data[:, 1]
        negatives = batch_data[:, 2]
        p_score, n_score = self.forward(users, positives, negatives)
        loss = self.BPRLoss(p_score, n_score)
        return loss

    def predict(self, users):

        user_embedding = self.user_embedding(users).view(-1, 1, 1, self.embedding_size)
        items_embedding = self.item_embedding.weight.view(1, -1, self.embedding_size, 1)
        scores = torch.matmul(user_embedding, items_embedding).squeeze()

        return scores
