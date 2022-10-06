import numpy as np
import torch
import torch.optim
import os

import configs
import backbone
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.crml import CRML
from pytorch_metric_learning import losses 

from io_utils import parse_args

def train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params):
    max_rec = 0 
    max_ap = 0

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        train_loss, train_rec, train_ap = model.train_loop(epoch, base_loader,  optimizer) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        test_loss, test_rec, test_ap = model.val_loop(val_loader)
        
        if test_rec[0] > max_rec : 
            print('best model for recall! save...') 
            max_rec = test_rec[0] 
            outfile = os.path.join(params.checkpoint_dir, 'best_model_rec.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
            
        if test_ap > max_ap : 
            print('best model for ap! save...')
            max_ap = test_ap
            outfile = os.path.join(params.checkpoint_dir, 'best_model_ap.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'CUB' or params.dataset == 'miniImagenet':
        base_file = configs.data_dir[params.dataset] + 'base.json' 
        val_file   = configs.data_dir[params.dataset] + 'val.json'

    image_size = 224
    params.stop_epoch = 600 # default
    
    base_model = lambda: backbone.DimReduce(backbone.ResNet18, 128)
    loss_fn = losses.MultiSimilarityLoss()
    
    if params.method == 'sft':
        num_samples_per_class = 5
        n_way = params.batch_size // num_samples_per_class
        train_few_shot_params = dict(n_way=n_way, n_support=0, n_query=num_samples_per_class)  # batch_size = n_support + n_query
        base_datamgr = SetDataManager(image_size, **train_few_shot_params)
        base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        test_few_shot_params = dict(n_way=params.test_n_way, n_support=0, n_query=params.n_query)
        val_datamgr = SetDataManager(image_size, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        
        model = BaselineTrain(base_model, params, loss_fn=loss_fn)

    elif params.method == 'crml':
        train_few_shot_params   = dict(n_way=params.train_n_way, n_support=params.n_shot)
        base_datamgr            = SetDataManager(image_size, n_query=params.n_query, **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )

        test_few_shot_params    = dict(n_way=params.test_n_way, n_support=params.n_shot)
        val_datamgr             = SetDataManager(image_size, n_query=params.n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False)
        
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        
        backbone.SimpleBlock.mtl = True
        backbone.ResNet.mtl = True
        backbone.DimReduction.mtl = True 
        model           = CRML(base_model, approx=True , loss_fn=loss_fn, **train_few_shot_params)
        
    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    params.checkpoint_dir = '%s/%s/%s' %(configs.save_dir, params.dataset, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method in ['baseline']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)
        
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    
    if params.pretrained is not None:
        pretrained_file = torch.load(params.pretrained)
        current_state = model.state_dict()
        current_state.update(pretrained_file['state'])
        model.load_state_dict(current_state)
        
        print('Loading ' + params.pretrained + ' at epoch ' + str(pretrained_file['epoch']))
        
    model = train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params)
