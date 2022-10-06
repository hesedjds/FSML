# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import os
import math
import torch
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler, ConditionSetDataset
from abc import abstractmethod
from torch.utils.data.sampler import Sampler


class NPairs(Sampler):
    def __init__(self, data_source, batch_size, num_batch, m=5):
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.m = m
        self.batch_size = batch_size
        self.class_idx = list(set(data_source.meta['image_labels']))
        self.num_batch = num_batch
        self.images_by_class = dict()
        for idx in self.class_idx:
            self.images_by_class[idx] = [i for i, e in enumerate(data_source.meta['image_labels']) if e == idx].copy()

    def __iter__(self):
        for _ in range(self.num_batch):
            selected_class = random.sample(self.class_idx, k=len(self.class_idx))
            example_indices = []

            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]
                new_ind = random.sample(img_ind_of_cls, k=min(self.m, len(img_ind_of_cls)))
                example_indices += new_ind

                if len(example_indices) >= self.batch_size:
                    break
            yield example_indices[:self.batch_size]

    def __len__(self):
        return self.num_batch


# Paired sampler "without replacement"
class NPairsWOR(Sampler):
    def __init__(self, data_source, class_pool, way, shot):
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.way = way
        self.shot = shot
        self.batch_size = way * shot
        self.class_pool = class_pool
        self.images_by_class = dict()
        for idx in self.class_pool:
            self.images_by_class[idx] = np.random.permutation([i for i, e in enumerate(data_source.meta['image_labels']) if e == idx]).copy()
            # print(idx, len(self.images_by_class[idx]))
        self.num_batch = math.ceil(max([len(self.images_by_class[i]) for i in self.images_by_class]) // shot)

    def __iter__(self):
        for i in range(self.num_batch):
            selected_class = random.sample(self.class_pool, k=self.way)
            example_indices = []

            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]
                if len(img_ind_of_cls) == 0:
                    continue

                new_ind = img_ind_of_cls[:self.shot]
                self.images_by_class[c] = img_ind_of_cls[self.shot:]
                example_indices += list(new_ind)

                # For the first (batch_size - 1) batch
                if len(example_indices) >= self.batch_size:
                    break

            if len(example_indices) != 0:
                yield example_indices
            else:
                pass

    def __len__(self):
        return self.num_batch

    
class NPairUse(Sampler): 
    def __init__(self, data_source, class_pool, shot, batch_size): 
        super(Sampler, self).__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.class_pool = class_pool 
        self.shot = shot
        self.images_by_class = dict()
        for idx in self.class_pool:
            self.images_by_class[idx] = np.random.permutation([i for i, e in enumerate(data_source.meta['image_labels']) if e == idx]).copy()
            # print(idx, len(self.images_by_class[idx]))
        self.num_batch = math.ceil(len(data_source.meta['image_labels']) / batch_size)
        
    def __iter__(self): 
        class_pool = np.random.permutation(self.class_pool)
        num_class = self.batch_size // self.shot 
        for i in range(self.num_batch): 
            example_indices = []
            start = i * num_class 
            end = (i+1) * num_class 
            selected_class = class_pool[start:end]
            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]
                example_indices += list(img_ind_of_cls)
                    
            yield example_indices 
    
    def __len__(self): 
        return self.num_batch
        
            
class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

    
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class AdaptAndRetreiveManager(DataManager):
    def __init__(self, image_size, adapt_way, n_shot, data_file, is_condition_set, aug, test):
        super(AdaptAndRetreiveManager, self).__init__()
        self.way = adapt_way
        self.shot = n_shot
        self.trans_loader = TransformLoader(image_size)
        self.transform = self.trans_loader.get_composed_transform(aug)
        self.is_condition_set = is_condition_set
        if self.is_condition_set:
            self.dataset = None
            self.condition_list = [os.path.join(data_file, f) for f in os.listdir(data_file) if
                              os.path.isfile(os.path.join(data_file, f)) and f[0] != '.']
            self.condition_list.sort()
            if test:
                idx = np.array([0, 1])
                self.condition_list = np.array(self.condition_list)[idx].tolist()
            else:
                raise NotImplementedError('Non-test option not supported for AdaptAndRetreiveManager. '
                                          'For train/val split, use AttributeDataManager.')
        else:
            self.dataset = SimpleDataset(data_file, self.transform)
            print(data_file, len(self.dataset), 'images')

    def get_data_loader(self):
        if self.is_condition_set:
            selected_condition_file = random.sample(self.condition_list, k=1)
            self.dataset = SimpleDataset(selected_condition_file[0], self.transform)
        class_all = list(set(self.dataset.meta['image_labels']))
        selected_class = random.sample(class_all, k=self.way)
        data_loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_sampler=NPairsWOR(self.dataset, class_pool=selected_class,
                                                                          way=self.way, shot=self.shot),
                                                  num_workers=12, pin_memory=True)
        return data_loader
    
    
class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug, is_ce=False, is_test=False): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform, is_ce=is_ce, is_test=is_test)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 16, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

    
class AttributeDataManager(DataManager): 
    def __init__(self, image_size, n_way, n_support, n_query, n_episode=100, train=False, val=False, test=False): 
        super(AttributeDataManager, self).__init__() 
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query 
        self.n_episode = n_episode 
        
        self.trans_loader = TransformLoader(image_size) 
        self.train = train 
        self.val = val
        self.test = test
        
    def get_data_loader(self, data_folder, aug): 
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = ConditionSetDataset(data_folder, self.n_way, self.batch_size, transform, self.n_episode, train=self.train, val=self.val, test=self.test) 
        data_loader_params = dict(batch_size=1, shuffle=True, num_workers=12, pin_memory=True) 
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params) 
        
        return data_loader