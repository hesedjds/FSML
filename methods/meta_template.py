import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils
from abc import abstractmethod

import copy 

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True, normalize_on_testing=False):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        # self.feature    = model_func
        # self.feat_dim   = self.feature.module.final_feat_dim
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test
        self.normalize_on_testing = normalize_on_testing 
        
    @abstractmethod
    def set_forward(self,x):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x):
        x    = x.cuda()
        
        x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        z_all       = self.feature.forward(x)
        z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        
        z_query     = z_all[:, self.n_support:]
                      
        return z_support, z_query

    def correct(self, scores, y=None):
        if y is None:
            y = np.repeat(range(self.n_way), self.n_query)
        else:
            y = y.cpu().numpy()

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y)
        return float(top1_correct), len(y)

    def encode_cls_label(self, y):
        d = {ni: indi for indi, ni in enumerate(set(y.tolist()))}
        y_cls = torch.LongTensor([d[ni.item()] for ni in y]).cuda()
        return y_cls

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10

        avg_loss=0
        acc_all = []
        rec_all = []
        ap_all = []
        for i, (x,_ ) in enumerate(train_loader):
            x = x.squeeze(0)
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss, scores, rec, ap = self.set_forward_loss( x )
            with torch.no_grad():
                correct_this, count_this = self.correct(scores)
                acc_all.append(correct_this / count_this * 100) 
                rec_all.append(rec)
                ap_all.append(ap)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()
        
            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
        
        acc_mean = torch.tensor(acc_all).mean().item()
        rec_all = np.asarray(rec_all) * 100
        rec_mean = np.mean(rec_all, axis=0)
        ap_mean = torch.tensor(ap_all).mean().item()
        
        return avg_loss / len(train_loader), acc_mean, rec_mean, ap_mean

    def val_loop(self, val_loader, record = None):
        with torch.no_grad():
            avg_loss = 0
            acc_all = []
            rec_all = []
            ap_all = []

            iter_num = len(val_loader)
            for i, (x,_) in enumerate(val_loader):
                x = x.squeeze(0)
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way  = x.size(0)
                loss, scores, rec, ap = self.set_forward_loss(x)
                correct_this, count_this = self.correct(scores)
                acc_all.append(correct_this/ count_this*100  )
                rec_all.append(rec)
                ap_all.append(ap)
                avg_loss = avg_loss + loss.item()

            acc_all  = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std  = np.std(acc_all)
            rec_all = np.asarray(rec_all) * 100
            rec_mean = np.mean(rec_all, axis=0)
            rec_std = np.std(rec_all, axis=0)
            ap_all = np.array(ap_all) * 100
            ap_mean = np.mean(ap_all)
            ap_std = np.std(ap_all)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
            for i in range(len(rec_mean)):
                print('%d Test Rec = %4.2f%% +- %4.2f%%' % (iter_num, rec_mean[i], 1.96 * rec_std[i]/np.sqrt(iter_num)))
            print('%d Test mAP = %4.2f%% +- %4.2f%%' % (iter_num, ap_mean, 1.96 * ap_std / np.sqrt(iter_num)))

        return avg_loss / len(val_loader), acc_mean, rec_mean, ap_mean

    def val_loop_adapt(self, val_loader, record = None): 
        avg_loss = 0
        acc_all = []
        rec_all = []
        ap_all = []

        iter_num = len(val_loader)
        for i, (x,_) in enumerate(val_loader):
            model_adapt = copy.deepcopy(self) 
            optimizer = torch.optim.SGD(model_adapt.parameters(), lr=0.01) 
            x = x.squeeze(0)
            model_adapt.n_query = x.size(1) - model_adapt.n_support
            if self.change_way:
                model_adapt.n_way  = x.size(0)

            model_adapt.train()
            for j in range(self.task_update_num):
                optimizer.zero_grad()
                loss, scores, rec, ap = model_adapt.set_forward_loss(x)
                loss.backward()
                optimizer.step()
            model_adapt.eval()
            
            with torch.no_grad():
                loss, scores, rec, ap = model_adapt.set_forward_loss(x) 
                correct_this, count_this = model_adapt.correct(scores)
            acc_all.append(correct_this/ count_this*100)
            rec_all.append(rec)
            ap_all.append(ap)
            avg_loss = avg_loss + loss.item()
            
            if i % 50 == 49: 
                print('%dth Test Acc = %4.2f%%' % (i + 1, np.asarray(acc_all).mean()))

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        rec_all = np.asarray(rec_all) * 100
        rec_mean = np.mean(rec_all, axis=0)
        rec_std = np.std(rec_all, axis=0)
        ap_all = np.array(ap_all) * 100
        ap_mean = np.mean(ap_all)
        ap_std = np.std(ap_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        for i in range(len(rec_mean)):
            print('%d Test Rec = %4.2f%% +- %4.2f%%' % (iter_num, rec_mean[i], 1.96 * rec_std[i]/np.sqrt(iter_num)))
        print('%d Test mAP = %4.2f%% +- %4.2f%%' % (iter_num, ap_mean, 1.96 * ap_std / np.sqrt(iter_num)))

        return avg_loss / len(val_loader), acc_mean, rec_mean, ap_mean
        
    
    def mean_std(self, all):
        mean = np.mean(all, axis=0)
        std = np.std(all, axis=0)
        return mean, std

    
