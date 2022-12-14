# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])
        
    
class SetDataset:
    def __init__(self, data_file, batch_size, transform, is_ce=False, is_test=False):
        self.data_file = data_file 
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        if is_ce: 
            new_label = np.arange(len(self.cl_list))
            one_hot_label = (self.meta['image_labels'] == np.expand_dims(np.unique(self.meta['image_labels']), 1))
            self.meta['image_labels'] = np.matmul(new_label, one_hot_label)
            self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False, drop_last=True)
        # print(self.data_file)
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( InfiniteDataLoader(sub_dataset, **sub_data_loader_params) )
            # print(len(self.sub_meta[cl]))

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
#         print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
        
    
class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

            
class ConditionSetDataset: 
    def __init__(self, data_folder, n_way, batch_size, transform, num_episode, train=False, val=False, test=False): 
        self.data_list = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and f[0] != '.'] 
        self.sub_dataloader = []
        self.num_episode = num_episode
        self.data_list.sort() 
        
        if train: 
            idx = np.array([2, 4 ,5])
            self.data_list = np.array(self.data_list)[idx].tolist()
        if val: 
            idx = np.array([3])
            self.data_list = np.array(self.data_list)[idx].tolist()
        if test:
            idx = np.array([0, 1])
            self.data_list = np.array(self.data_list)[idx].tolist()
        
        for data in self.data_list:
            sub_dataset = SetDataset(data, batch_size, transform=transform)
            sub_sampler = EpisodicBatchSampler(len(sub_dataset.cl_list), n_way, num_episode)
            sub_data_loader_params = dict(batch_sampler=sub_sampler,
                                          num_workers=0,
                                          pin_memory=False) 
#             self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))
            self.sub_dataloader.append(InfiniteDataLoader(sub_dataset, **sub_data_loader_params))
            
    def __getitem__(self, i): 
        i = i % len(self.data_list)
#         print(self.sub_dataloader[i], i)
        data = next(iter(self.sub_dataloader[i]))
        
        return data
    
    def __len__(self): 
        return self.num_episode