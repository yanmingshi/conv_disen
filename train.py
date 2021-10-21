#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author: yanms
# @Date  : 2021/8/12 17:59
# @Desc  :
import argparse
import copy
import random
import time
from tqdm import tqdm

from torch import optim
from torch.utils.data import DataLoader
from logger import Logger
from metrics import metrics_dict

from data_set import DataSetGenerator
from mf import MF
from model import CNNNet
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from wide import Wide
from wide1 import Wide1
from wide2 import Wide2
from wide3 import Wide3
from wide4 import Wide4
from wide5 import Wide5

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数
torch.cuda.manual_seed(SEED)  # 为GPU设置种子用于生成随机数
torch.cuda.manual_seed_all(SEED)  # 为多个GPU设置种子用于生成随机数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TIME = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())


class Trainer(object):
    def __init__(self, model, dataset: DataSetGenerator, args):
        self.model = model.to(device)
        self.dataset = dataset
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.evaluate_batch_size = args.evaluate_batch_size
        self.min_epoch = args.min_epoch
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.optimizer = self.get_optimizer(self.model)
        self.gt_length = np.array(dataset.validate_items_length)
        self.writer = SummaryWriter('./log/' + self.model_name + TIME)
        self.logging = Logger(args.model_name, level='debug').logger

    def get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def train_model(self):
        train_loader = DataLoader(dataset=self.dataset.train_dataset, batch_size=self.batch_size, shuffle=True)
        min_loss = 10  # 用来保存最好的模型
        best_recall, best_ndcg, best_epoch = 0.0, 0.0, 0
        for epoch in range(self.epochs):
            total_loss = 0.0
            self.model.train()
            start_time = time.time()
            train_data_iter = (
                tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"\033[1;35mTrain {epoch:>5}\033[0m"
                )
            )
            batch_no = 0
            for batch_index, batch_data in train_data_iter:
                batch_data = batch_data.to(device)
                self.optimizer.zero_grad()
                loss = self.model.calculate(batch_data)
                loss.backward()
                self.optimizer.step()
                batch_no = batch_index
                total_loss += loss.item()

            total_loss = total_loss / (batch_no + 1)
            # 记录loss到tensorboard可视化
            self.writer.add_scalar('training loss', total_loss, epoch+1)
            epoch_time = time.time() - start_time
            self.logging.info('epoch %d %.2fs train loss is [%.4f] ' % (epoch + 1, epoch_time, total_loss))

            # evaluate
            metric_dict = self.evaluate(epoch, self.evaluate_batch_size)
            recall, ndcg = metric_dict['recall@' + str(self.topk)], metric_dict['ndcg@' + str(self.topk)]
            if epoch > self.min_epoch and recall > best_recall:
                best_recall, best_ndcg, best_epoch = recall, ndcg, epoch
                best_model = self.model
                # 保存最好的模型
                self.save_model(best_model)
        self.logging.info(f"training end, best iteration %d, results: recall@{self.topk}: %s, ndgc@{self.topk}: %s" %
                     (best_epoch+1, best_recall, best_ndcg))

    @torch.no_grad()
    def evaluate(self, epoch, batch_size):
        data_loader = DataLoader(dataset=self.dataset.validate_dataset, batch_size=batch_size)
        self.model.eval()
        start_time = time.time()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_items
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(device)
            scores = self.model.predict(batch_data)
            # 替换训练集中使用过的item为无穷小
            for user in batch_data:
                items = train_items[user]
                user_score = scores[(user-batch_index*batch_size)]
                user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, self.topk, dim=-1)
                gt_items = self.dataset.validate_items[user]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                # mask = (topk_idx - gt_items == 0)
                # mask = torch.sum(mask, dim=0) == 1
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, self.gt_length, epoch)
        epoch_time = time.time() - start_time
        self.logging.info(f"evaluator %d cost time %.2fs, result: %s " % (epoch, epoch_time, metric_dict.__str__()))
        return metric_dict

    def calculate_result(self, topk_list, gt_len, epoch):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for metric, value in zip(self.metrics, result_list):
            key = '{}@{}'.format(metric, self.topk)
            metric_dict[key] = np.round(value[self.topk - 1], 4)
            self.writer.add_scalar('evaluate ' + metric, metric_dict[key], epoch + 1)
        return metric_dict

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path + self.model_name + TIME + '.pth')



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set args', add_help=False)

    parser.add_argument('--embedding_size', type=int, default=32, help='')
    parser.add_argument('--behavior_count', type=int, default=4, help='')
    parser.add_argument('--layers', type=int, default=3, help='')
    parser.add_argument('--reg_weight', type=int, default=1e-4, help='')
    parser.add_argument('--lambda_', type=int, default=0.7, help='')
    parser.add_argument('--if_load_model', type=bool, default=False, help='')
    parser.add_argument('--model_full_name', type=str, default='./check_point/CNN4Rec2021-08-27 10_02_19.pth', help='')

    parser.add_argument('--topk', type=str, default=20, help='')
    parser.add_argument('--metrics', type=str, default=['recall', 'ndcg'], help='')
    parser.add_argument('--lr', type=int, default=0.001, help='')
    parser.add_argument('--decay', type=float, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--evaluate_batch_size', type=int, default=128, help='')
    parser.add_argument('--min_epoch', type=str, default=5, help='')
    parser.add_argument('--epochs', type=str, default=500, help='')
    parser.add_argument('--model_path', type=str, default='./check_point/', help='')
    parser.add_argument('--model_name', type=str, default='CNN4Rec', help='')

    parser.add_argument('--data_path', type=str, default='./data/Amazon-Book/', help='')

    args = parser.parse_args()

    args.device = device
    dataset = DataSetGenerator(args.data_path)
    # model = CNNNet(args)
    # model = Wide(args)
    # model = Wide1(args)
    # model = Wide2(args)
    # model = MF(args)
    # model = Wide3(args)
    # model = Wide4(args)
    model = Wide5(args)
    trainer = Trainer(model, dataset, args)

    trainer.logging.info(args.__str__())
    trainer.logging.info(model)

    trainer.train_model()
    # trainer.evaluate(12, 128)
