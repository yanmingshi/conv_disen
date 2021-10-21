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


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class Wide5(nn.Module):

    def __init__(self, args):
        super(Wide5, self).__init__()
        self.user_num = 52643
        self.item_num = 91599
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.reg_weight = args.reg_weight
        self.BPRLoss = BPRLoss()
        self.EmbLoss = EmbLoss()
        self.if_load_model = args.if_load_model

        self.feature = nn.Conv2d(1, 16, kernel_size=(1, 1))

        self.linear = nn.Linear(16, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if self.if_load_model:
            parameters = torch.load(self.model_full_name)
            self.load_state_dict(parameters)
        else:
            if isinstance(module, nn.Embedding):
                torch.nn.init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.weight.data, 1)
                
                
    def get_features(self, embedding):
        return torch.softmax(self.feature(embedding), dim=-1)


    def forward(self, users, positives, negatives):
        users_embedding = self.user_embedding(users).view(-1, 1, 1, self.embedding_size)
        positives_embedding = self.item_embedding(positives).view(-1, 1, 1, self.embedding_size)
        negatives_embedding = self.item_embedding(negatives).view(-1, 1, 1, self.embedding_size)
        user_features = self.get_features(users_embedding).squeeze()
        p_features = self.get_features(positives_embedding).squeeze()
        n_features = self.get_features(negatives_embedding).squeeze()

        p_score = torch.mul(user_features, p_features).sum(dim=-1)
        p_score = self.linear(p_score)
        n_score = torch.mul(user_features, n_features).sum(dim=-1)
        n_score = self.linear(n_score)
        return p_score, n_score

    def calculate(self, batch_data):
        users = batch_data[:, 0]
        positives = batch_data[:, 1]
        negatives = batch_data[:, 2]
        p_score, n_score = self.forward(users, positives, negatives)
        loss = self.BPRLoss(p_score, n_score) + self.reg_weight * self.EmbLoss(self.user_embedding(users), self.item_embedding(positives), self.item_embedding(negatives))
        return loss

    def predict(self, users):
        users_embedding = self.user_embedding(users).view(-1, 1, 1, self.embedding_size)
        items_embedding = self.item_embedding.weight.view(-1, 1, 1, self.embedding_size)
        u_features = self.get_features(users_embedding)
        i_features = self.get_features(items_embedding)
        u_shape = u_features.shape
        i_shape = i_features.shape
        u_features = u_features.view(u_shape[0], 1, u_shape[1], u_shape[3])
        i_features = i_features.view(1, i_shape[0], i_shape[1], i_shape[3])
        score = torch.mul(u_features, i_features).sum(dim=-1)
        score = self.linear(score).squeeze()
        return score
