import time
import torch.nn as nn
import torch.utils.data.sampler

import configs
import backbone
from data.datamgr import AdaptAndRetreiveManager
from methods.baselinetrain import BaselineTrain
from methods.crml import CRML

from pytorch_metric_learning import losses 
from io_utils import parse_args

if __name__ == '__main__':
    params = parse_args('test')

    iter_num = params.iter_num
    base_model = lambda: backbone.DimReduce(backbone.ResNet18, 128)
    loss_fn = losses.MultiSimilarityLoss(alpha=2.0, beta=4.0)
    
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    if params.method == 'sft':
        model           = BaselineTrain(base_model, params, loss_fn=loss_fn )
    elif params.method == 'crml':
        backbone.SimpleBlock.mtl = True
        backbone.ResNet.mtl = True
        backbone.DimReduction.mtl = True 
        model           = CRML(base_model, approx = True , loss_fn=loss_fn, **few_shot_params )
        model.train_lr = params.crml_lr
    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    
    if params.path is not None: 
        tmp = torch.load(params.path)
        print('loading ' + params.path.split('/')[-1] + ' at epoch ' + str(tmp['epoch']))
        model.load_state_dict(tmp['state'])
    else:
        raise ValueError('path of the trained model to evaluate should be specifed')

    split = 'novel'
    image_size = 224
    
    if params.dataset == 'Fashion':
        base_folder = configs.data_dir['Fashion']
        datamgr = AdaptAndRetreiveManager(image_size, adapt_way=params.test_n_way, n_shot=params.n_shot,
                                          data_file=base_folder, is_condition_set=True, aug=False, test=True)
    else:
        loadfile = configs.data_dir[params.dataset] + split + '.json'
        datamgr = AdaptAndRetreiveManager(image_size, adapt_way=params.test_n_way, n_shot=params.n_shot,
                                          data_file=loadfile, is_condition_set=False, aug=False, test=True)

    model.task_update_num = 10
    model.eval()

    model.adapt_and_test_all(datamgr, iter_num)


