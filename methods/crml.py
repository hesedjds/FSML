# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 
# metric MAMl 

import copy
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from utils import recall, mAP


class CRML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, loss_fn, approx = False):
        super(CRML, self).__init__( model_func,  n_way, n_support, change_way = False)

        self.loss_fn = loss_fn 
        
        self.n_task     = 4
        self.task_update_num = 10
        self.train_lr = 1.0
        self.approx = approx #first order approx.        
    
    def forward_feat(self, x): 
        out = self.feature.forward(x) 
        return out

    def set_forward(self, x):
        x = x.cuda()
        x_var = Variable(x)
        n_way = x_var.size(0)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( n_way ), self.n_support ) )).cuda() #label for support data
        fast_parameters = [p for p in self.parameters() if p.requires_grad] #the first gradient calcuated in line 45 is based on original weight
        
        for weight in fast_parameters:
            weight.fast = None
        self.zero_grad()
        
        for task_step in range(self.task_update_num):
            feat = self.forward_feat(x_a_i)
            feat = F.normalize(feat, p=2, dim=-1)
            set_loss = self.loss_fn(feat, y_a_i) 

            if set_loss == 0:
                feat = self.forward_feat(x_b_i)
                return 0, feat
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate([p for p in self.parameters() if p.requires_grad]):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None: 
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

        feat = self.forward_feat(x_b_i)
        return feat

    def set_forward_loss(self, x):
        n_way = x.size(0)
        feat = self.set_forward(x)
        
        y_b_i = torch.from_numpy(np.repeat(range( n_way ), self.n_query)).to(x.device)
        feat = F.normalize(feat, p=2, dim=-1)
        loss = self.loss_fn(feat, y_b_i)
        
        with torch.no_grad(): 
            rec = recall(feat, y_b_i, K=[1, 2, 4, 8])
            map = mAP(feat, y_b_i)
            
        return loss, rec, map

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        rec_all = []
        map_all = []
        optimizer.zero_grad()

        #train
        for i, (x, _) in enumerate(train_loader):
            x = x.squeeze(0)
            x = x.cuda()
            self.n_query = x.size(1) - self.n_support
            
            loss, rec, map = self.set_forward_loss(x)
            try:
                avg_loss = avg_loss+loss.item()
            except:
                avg_loss = avg_loss
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            
            with torch.no_grad():
                rec_all.append(rec)
                map_all.append(map)
                
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
        rec_all = np.asarray(rec_all) * 100
        map_all = np.asarray(map_all) * 100
        rec_mean = np.mean(rec_all, axis=0)       
        map_mean = np.mean(map_all, axis=0)       
        
        return avg_loss / len(train_loader), rec_mean, map_mean

    def val_loop(self, test_loader, return_std = False): #overwrite parrent function
        rec_all = []
        map_all = []
        avg_loss = 0.0 
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            x = x.squeeze(0)
            x = x.cuda()
            self.n_query = x.size(1) - self.n_support
            
            loss, rec, map = self.set_forward_loss(x)
            rec_all.append(rec)
            map_all.append(map)
            avg_loss = avg_loss + loss.item()

        rec_all = np.asarray(rec_all) * 100
        rec_mean = np.mean(rec_all, axis=0)
        rec_std = np.std(rec_all, axis=0)
        map_all = np.asarray(map_all) * 100
        map_mean = np.mean(map_all, axis=0)
        map_std = np.std(map_all, axis=0)
        for i in range(len(rec_mean)):
            print('%d Test Rec = %4.2f%% +- %4.2f%%' % (iter_num, rec_mean[i], 1.96 * rec_std[i]/np.sqrt(iter_num)))
        print('%d Test mAP = %4.2f%% +- %4.2f%%' % (iter_num, map_mean, 1.96 * map_std/np.sqrt(iter_num)))

        return avg_loss / len(test_loader), rec_mean, map_mean

    def adapt_loop(self, x, y):
        self.optimizer_adapt.zero_grad()
        embeddings = self.feature_adapt.forward(x)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        loss = self.loss_fn(embeddings, y)
        loss.backward()
        self.optimizer_adapt.step()
        rec = recall(embeddings, y, K=[1])[0]
        return rec        
        
    def adapt_loop_sop(self, x, y):
        self.optimizer_adapt.zero_grad()
        embeddings = self.feature_adapt.forward(x)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        loss = self.loss_fn(embeddings, y)
        loss.backward()
        self.optimizer_adapt.step()
        return embeddings

    def adapt_and_test_all(self, datamgr, iter_num):
        assert self.task_update_num > 0, "Specify adapting iteration."
        print('Adapting {} times with {} way {} shot'.format(self.task_update_num, datamgr.way, datamgr.shot))

        print_freq = 5
        K = [1, 2, 4, 8]
        _rec_all, rec_all, _map_all, map_all, train_rec_all = [], [], [], [], []
        
        for _iter in range(iter_num):
            loader = datamgr.get_data_loader()
            self.feature_adapt = copy.deepcopy(self.feature)
            self.optimizer_adapt = torch.optim.Adam(self.feature_adapt.parameters(), lr=self.train_lr)

            for weight in self.feature_adapt.parameters():
                weight.fast = None
            self.zero_grad()
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
                labels_all = torch.cat(labels_all).cpu()
                eval_num_total = labels_all.size(0)

                # evaluate the adapted model
                embeddings_all = torch.cat(embeddings_all).cpu()
                rec = recall(embeddings_all, labels_all, K=K)
                rec_all.append(rec)
                map = mAP(embeddings_all, labels_all)
                map_all.append(map)

                # evaluate the non-adapted model
                _embeddings_all = torch.cat(_embeddings_all).cpu()
                _rec = recall(_embeddings_all, labels_all, K=K)
                _rec_all.append(_rec)
                _map = mAP(_embeddings_all, labels_all)
                _map_all.append(_map)

            if _iter % print_freq == 0:
                print('[iter %d] [Eval set] %d images out of %d tested' % (_iter, eval_num_total, len(loader.dataset)))
                for k, _r, r in zip(K, _rec, rec):
                    print('[iter %d] [Eval set] Recall@%d: %4.2f%% -> %4.2f%%' % (_iter, k, _r * 100, r * 100))
                print('[iter %d] [Eval set] mAP: %4.2f%% -> %4.2f%%' % (_iter, _map * 100, map * 100))
                rec_mean, rec_std = self.mean_std(np.asarray(rec_all) * 100)
                map_mean, map_std = self.mean_std(np.asarray(map_all) * 100)
                print('[Iter %d][Eval set] Recall@1 until now: %4.2f%% +- %4.2f%%' % (_iter, rec_mean[0], 1.96 * rec_std[0] / np.sqrt(_iter)))
                print('[Iter %d][Eval set] mAP until now: %4.2f%% +- %4.2f%%' % (_iter, map_mean, 1.96 * map_std / np.sqrt(_iter)))

        train_rec_mean = np.mean(np.asarray(train_rec_all) * 100, axis=0)
        rec_mean, rec_std = self.mean_std(np.asarray(rec_all) * 100)
        map_mean, map_std = self.mean_std(np.asarray(map_all) * 100)
        _rec_mean, _rec_std = self.mean_std(np.asarray(_rec_all) * 100)
        _map_mean, _map_std = self.mean_std(np.asarray(_map_all) * 100)

        print('\n====================================================')
        print('[Averaged over %d times]' % iter_num)
        print('[Adaptation set] Recall@1: ', train_rec_mean)
        for k, _r, _r_std, r, r_std in zip(K, _rec_mean, _rec_std, rec_mean, rec_std):
            print('Recall@%d: %4.2f%% +- %4.2f%% -> %4.2f%% +- %4.2f%%'
                  % (k, _r, 1.96 * _r_std / np.sqrt(iter_num), r, 1.96 * r_std / np.sqrt(iter_num)))
        print('mAP: %4.2f%% +- %4.2f%% -> %4.2f%% +- %4.2f%%' % (_map_mean, 1.96 * _map_std / np.sqrt(iter_num), map_mean, 1.96 * map_std / np.sqrt(iter_num)))
        print('====================================================')
        return rec_mean, map_mean
