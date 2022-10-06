import backbone
import utils

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import recall, mAP
from collections import OrderedDict
import torch.nn.functional as F


class BaselineTrain(nn.Module):
    def __init__(self, model_func, params, loss_fn, loss_type='softmax', change_way=True, task_update_num=0):
        super(BaselineTrain, self).__init__()
        self.feature    = model_func()
        self.n_support = params.n_shot
        self.change_way = change_way
        self.train_n_way = params.train_n_way
        self.test_n_way = params.test_n_way  # n_way is used only for meta-testing, thus I hard-code n_way as test_n_way
        self.loss_fn = loss_fn
        self.task_update_num = task_update_num

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        loss_all = 0
        rec_all = []
        map_all = []

        for i, (x, y) in enumerate(train_loader):
            cwh = x.shape[-3:]
            x, y = x.view(-1, *cwh), y.view(-1)
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            embeddings = self.feature.forward(x)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            loss = self.loss_fn(embeddings, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_all = loss_all + loss.item()
                rec = recall(embeddings, y, K=[1, 2, 4, 8])
                rec_all.append(rec)
                ap = mAP(embeddings, y) 
                map_all.append(ap)

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.
                      format(epoch, i, len(train_loader), loss_all / float(i + 1)))

        rec_mean = np.mean(rec_all, axis=0) * 100
        map_mean = np.mean(map_all) * 100 
        loss_mean = loss_all / float(len(train_loader))
        return loss_mean, rec_mean, map_mean

    def val_loop(self, val_loader):
        rec_all = []
        map_all = []
        loss_all = 0

        with torch.no_grad():
            iter_num = len(val_loader)
            for i, (x, y) in enumerate(val_loader):
                cwh = x.shape[-3:]
                x, y = x.view(-1, *cwh), y.view(-1)
                x, y = x.cuda(), y.cuda()
                embeddings = self.feature.forward(x)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                loss = self.loss_fn(embeddings, y)
                loss_all = loss_all + loss.item()
                rec = recall(embeddings, y, K=[1, 2, 4, 8])
                
                rec_all.append(rec)
                ap = mAP(embeddings, y)
                map_all.append(ap)

            rec_all = np.asarray(rec_all) * 100
            rec_mean = np.mean(rec_all, axis=0)
            rec_std = np.std(rec_all, axis=0)
            
            map_all = np.asarray(map_all) * 100 
            map_mean = np.mean(map_all)
            map_std = np.std(map_all)
            print(rec_mean)
            for i in range(len(rec_mean)):
                print('%d Test Rec = %4.2f%% +- %4.2f%%' % (iter_num, rec_mean[i], 1.96 * rec_std[i] / np.sqrt(iter_num)))
            print('%d Test mAP = %4.2f%% +- %4.2f%%' % (iter_num, map_mean, 1.96 * map_std / np.sqrt(iter_num)))
            
        return loss_all / len(val_loader), rec_mean, map_mean

    def adapt_loop(self, x, y):
        self.optimizer_adapt.zero_grad()
        embeddings = self.feature_adapt.forward(x)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        loss = self.loss_fn(embeddings, y)
        loss.backward()
        self.optimizer_adapt.step()
        rec = recall(embeddings, y, K=[1])[0]
        return rec

    def mean_std(self, all):
        mean = np.mean(all, axis=0)
        std = np.std(all, axis=0)
        return mean, std
                

    def adapt_and_test_all(self, datamgr, iter_num):
        assert self.task_update_num > 0, "Specify adapting iteration."
        print('Adapting {} times with {} way {} shot'.format(self.task_update_num, datamgr.way, datamgr.shot))

        print_freq = 5
        K = [1, 2, 4, 8]
        _rec_all, rec_all, _map_all, map_all, train_rec_all = [], [], [], [], []
        acc_all = []
        for _iter in range(iter_num):
            loader = datamgr.get_data_loader()
            self.feature_adapt = copy.deepcopy(self.feature)
            self.optimizer_adapt = torch.optim.Adam([
                {'params': self.feature_adapt.base.parameters(), 'lr': 0.0005},
                {'params': self.feature_adapt.dim_reduction.parameters(), 'lr': 0.005}
            ], lr=0.0005)

            _embeddings_all, embeddings_all, labels_all = [], [], []
            for i, (x, y) in enumerate(loader):
                x, y = x.cuda(), y.cuda()
                # adapting with the first N-way K-shot
                if i == 0:
                    train_rec_str = '\n[iter {}] [Adaptation set] Recall@1: '.format(_iter)
                    self.feature_adapt.train()
                    train_rec = []
                    for adapt_iter in range(self.task_update_num):
                        rec = self.adapt_loop(x, y)
                        train_rec_str += '%4.2f  ' % (rec * 100)
                        train_rec.append(round(rec, 2))
                    train_rec_all.append(train_rec)
                    if _iter % print_freq == 0:
                        print(train_rec_str)
                # evaluating the rest N-way full-prediction data
                else:
                    self.feature_adapt.train()
                    with torch.no_grad():
                        labels_all.append(y.data)

                        # evaluate the adapted model
                        embeddings = self.feature_adapt.forward(x)
                        embeddings = F.normalize(embeddings, p=2, dim=-1)
                        embeddings_all.append(embeddings.data)

                        # evaluate the non-adapted model
                        _embeddings = self.feature.forward(x)
                        _embeddings = F.normalize(_embeddings, p=2, dim=-1)
                        _embeddings_all.append(_embeddings.data)

            with torch.no_grad():
                labels_all = torch.cat(labels_all)
                eval_num_total = labels_all.size(0)

                # evaluate the adapted model
                embeddings_all = torch.cat(embeddings_all)
                embeddings_all = F.normalize(embeddings_all, p=2, dim=-1).cpu()
                labels_all = labels_all.cpu()
                rec = recall(embeddings_all, labels_all, K=K)
                ap = mAP(embeddings_all, labels_all)
#                 
                rec_all.append(rec)
                map_all.append(ap)

                # evaluate the non-adapted model
                _embeddings_all = torch.cat(_embeddings_all).cpu()
                _rec = recall(_embeddings_all, labels_all, K=K)
                _ap = mAP(_embeddings_all, labels_all)
                _rec_all.append(_rec)
                _map_all.append(_ap)

            if _iter % print_freq == 0:
                print('[iter %d] [Eval set] %d images out of %d tested' % (_iter, eval_num_total, len(loader.dataset)))
                for k, _r, r in zip(K, _rec, rec):
                    print('[iter %d] [Eval set] Recall@%d: %4.2f%% -> %4.2f%%' % (_iter, k, _r * 100, r * 100))
                _rec_mean, _ = self.mean_std(np.asarray(_rec_all) * 100)
                rec_mean, _ = self.mean_std(np.asarray(rec_all) * 100)
                _map_mean, _ = self.mean_std(np.asarray(_map_all) * 100) 
                map_mean, _ = self.mean_std(np.asarray(map_all) * 100) 
                
                print('Average Recall@1 until now: %4.2f%% -> %4.2f%%' % (_rec_mean[0], rec_mean[0]))
                print('Average mAP until now: %4.2f%% -> %4.2f%%' % (_map_mean, map_mean))

        train_rec_mean = np.mean(np.asarray(train_rec_all) * 100, axis=0)
        rec_mean, rec_std = self.mean_std(np.asarray(rec_all) * 100)
        map_mean, map_std = self.mean_std(np.asarray(map_all) * 100)
        _rec_mean, _rec_std = self.mean_std(np.asarray(_rec_all) * 100)
        rec_mean, rec_std = self.mean_std(np.asarray(rec_all) * 100)
        _map_mean, _map_std = self.mean_std(np.asarray(_map_all) * 100)
        map_mean, map_std = self.mean_std(np.asarray(map_all) * 100)

        print('\n====================================================')
        print('[Averaged over %d times]' % iter_num)
        print('[Adaptation set] Recall@1: ', train_rec_mean)
        for k, _r, _r_std, r, r_std in zip(K, _rec_mean, _rec_std, rec_mean, rec_std):
            print('Recall@%d: %4.2f%% +- %4.2f%% -> %4.2f%% +- %4.2f%%'
                  % (k, _r, 1.96 * _r_std / np.sqrt(iter_num), r, 1.96 * r_std / np.sqrt(iter_num)))
        print('mAP: %4.2f%% +- %4.2f%% -> %4.2f%% +- %4.2f%%' 
              % (_map_mean, 1.96 * _map_std / np.sqrt(iter_num), map_mean, 1.96 * map_std / np.sqrt(iter_num)))
        print('====================================================')
        return rec_mean, map_mean
