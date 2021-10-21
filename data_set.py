#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_set.py
# @Author: yanms
# @Date  : 2021/8/5 16:46
# @Desc  :
import random
import time

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


def _data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     划分比例
    :param shuffle:   是否打乱
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def data_process(file_path):
    with open(file_path + 'train.txt', encoding='utf-8') as rf, \
            open(file_path + 'test.txt', encoding='utf-8') as rft, \
            open(file_path + 'train_data.txt', 'w', encoding='utf-8') as wf, \
            open(file_path + 'train_items.txt', 'w', encoding='utf-8') as wt, \
            open(file_path + 'validate_items.txt', 'w', encoding='utf-8') as wv:
        # data = rf.readlines()
        # for line in data:
        #     line = line.strip('\n').split(' ')
        #     user = line[0]
        #     items = line[1:]
            # train_item, validate_item = _data_split(items, 0.8)
            # for i in train_item:
            #     wf.write(user + ' ' + i + '\n')
            # wt.write(' '.join(train_item) + '\n')
            # wv.write(' '.join(validate_item) + '\n')

        train = rf.readlines()
        test = rft.readlines()
        for line in train:
            line = line.strip('\n').split(' ')
            user = line[0]
            items = line[1:]
            for i in items:
                wf.write(user + ' ' + i + '\n')
            wt.write(' '.join(items) + '\n')

        for test_line in test:
            test_line = test_line.strip('\n').split(' ')
            test_user = test_line[0]
            test_item = test_line[1:]
            wv.write(' '.join(test_item) + '\n')


# class DataGenerator(object):
#     def __init__(self, file_path, ratio):
#         super(DataGenerator, self).__init__()
#         with open(file_path + 'train.txt', encoding='utf-8') as f:
#             datas = f.readlines()
#             train_data = []
#             validate_data = []
#             all_items = []
#             train_items = []
#             validate_items = []
#             for line in datas:
#                 if len(line) > 0:
#                     line = line.strip('\n').split(' ')
#                     user = line[0]
#                     items = line[1:]
#                     all_items.append(list(map(eval, items)))
#                     train_item, validate_item = _data_split(items, ratio)
#                     train_one = [user]
#                     train_one.extend(train_item)
#                     train_data.append(list(map(eval, train_one)))
#                     validate_one = [user]
#                     validate_one.extend(validate_item)
#                     validate_data.append(list(map(eval, validate_one)))
#                     train_items.append(list(map(eval, train_item)))
#                     validate_items.append(list(map(eval, validate_item)))
#             self.train_data = train_data
#             self.validate_data = validate_data
#             self.all_items = all_items
#             self.train_items = train_items
#             self.validate_items = validate_items
#             self.validate_items_length = [len(x) for x in validate_items]
#         with open(file_path + 'test.txt', encoding='utf-8') as f:
#             test_data = []
#             for line in datas:
#                 if len(line) > 0:
#                     line = line.strip('\n').split(' ')
#                     test_data.append(list(map(eval, line)))
#             self.test_data = test_data


class DataSet(Dataset):
    def __init__(self, data=None, train_items=None, type='valid'):
        self.user_count = 52643
        self.item_count = 91599
        self.data = data
        self.train_items = train_items
        self.type = type
        self.sample_items = set([x for x in range(self.item_count)])

    def __getitem__(self, idx):
        if self.type.lower() == 'train':
            # 生成正负样本对
            user = int(self.data[idx][0])
            positive = self.data[idx][1]
            negative = random.randint(0, self.item_count - 1)
            while np.isin(negative, self.train_items[user]):
                negative = random.randint(0, self.item_count - 1)
            return np.array([user, positive, negative])
        else:
            return idx

    def __len__(self):
        if self.type.lower() == 'train':
            return len(self.data)
        else:
            return self.user_count


class DataSetGenerator(object):
    """
    加载训练集、验证集和测试集
    :return:
    """
    def __init__(self, file_path):
        train = pd.read_table(file_path + 'train_data.txt', sep=' ', header=None)
        train = train.to_numpy()

        train_items = []
        with open(file_path + 'train_items.txt', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                if len(line) > 0:
                    line = line.strip('\n').strip().split(' ')
                    train_items.append(list(map(int, line)))
        validate_items = []
        with open(file_path + 'validate_items.txt', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                if len(line) > 0:
                    line = line.strip('\n').strip().split(' ')
                    validate_items.append(list(map(int, line)))
        test = []
        with open(file_path + 'test.txt', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                if len(line) > 0:
                    line = line.strip('\n').strip().split(' ')
                    test.append(list(map(int, line)))
        self.train_data_list = train
        self.train_dataset = DataSet(data=train.tolist(), train_items=train_items, type='train')
        self.validate_dataset = DataSet()
        self.test_dataset = DataSet()
        self.train_items = train_items
        self.validate_items = validate_items
        self.validate_items_length = [len(x) for x in validate_items]


if __name__ == '__main__':
    type = 'train'
    file_path = './data/amazon-book/'
    # file_path = './data/gowalla/'
    # ratio = [0.8, 0.2]
    # #
    # data_generator = DataGenerator(file_path, 0.8)
    # train = DataSet(data_generator, 'validate')
    # data_loader = DataLoader(dataset=train, batch_size=3, shuffle=False)
    # for index, item in enumerate(data_loader):
    #     print(index, '-----', item)
    # data_process(file_path)
    dataset = DataSetGenerator(file_path)
    # train_loader = DataLoader(dataset=dataset.train_dataset, batch_size=3, shuffle=True)
    # for index, item in enumerate(train_loader):
    #     print(index, '-----', item)
    # valid_loader = DataLoader(dataset=dataset.validate_dataset, batch_size=3, shuffle=False)
    # for index, item in enumerate(valid_loader):
    #     print(index, '-----', item)
    # test_loader = DataLoader(dataset=dataset.test_dataset, batch_size=3, shuffle=False)
    # for index, item in enumerate(data_loader):
    #     print(index, '-----', item)
    # start = time.time()
    # data_loader = DataLoader(dataset=dataset.train_dataset, batch_size=512, shuffle=True)
    # iter_data = (
    #     tqdm(
    #         enumerate(data_loader),
    #         total=len(data_loader),
    #         desc=f"\033[1;35mEvaluate \033[0m"
    #     )
    # )
    # for batch_index, batch_data in iter_data:
    #     pass
        # print(batch_index, '---', batch_data)

    # print(f'end: %.4fs' % (time.time()-start))