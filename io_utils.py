import numpy as np
import os
import glob
import argparse
import backbone

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char/Fashion/sop/sop_domain')
    parser.add_argument('--method'      , default='crml',   help='sft/crml') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--batch_size', default=300, type=int, help='batch size only works for baseline method')
    parser.add_argument('--n_query'       , default=16, type=int, help='K-shot of query for meta-training')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained backbone model to training ours')
    parser.add_argument('--crml_lr', default=0.025, type=float, help='Learning rate for MTL inner')
    if script == 'train':
        # parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=20, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate for meta-training')
    elif script == 'test':
        parser.add_argument('--path', default=None, type=str, help='Model path to test')
        parser.add_argument('--iter_num', default=200, type=int, help='Iteration number for meta-testing episodes')
    else:
        raise ValueError('Unknown script')
        
          
    return parser.parse_args()