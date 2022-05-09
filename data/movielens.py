import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader


class Movielens1MbaseDataset(Dataset):
    """
    Load a base Movielens Dataset 
    """
    def __init__(self, dataset_name, df, description, device):
        super(Movielens1MbaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)
        self.name2array = {name: torch.from_numpy(np.array(list(df[name])).reshape([self.length, -1])).to(device) \
                                        for name in df.columns}
        self.format(description)
        self.features = [name for name in df.columns if name != 'rating']
        self.label = 'rating'

    def format(self, description):
        for name, size, type in description:
            if type == 'spr' or type == 'seq':
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type == 'ctn':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type == 'label':
                pass
            else:
                raise ValueError('unkwon type {}'.format(type))
                
    def __getitem__(self, index):
        return {name: self.name2array[name][index] for name in self.features}, \
                self.name2array[self.label][index].squeeze()

    def __len__(self):
        return self.length


class MovieLens1MColdStartDataLoader(object):
    """
    Load all splitted MovieLens 1M Dataset for cold start setting

    :param dataset_path: MovieLens dataset path
    """

    def __init__(self, dataset_name, dataset_path, device, bsz=32, shuffle=True):
        assert os.path.exists(dataset_path), '{} does not exist'.format(dataset_path)
        with open(dataset_path, 'rb+') as f:
            data = pickle.load(f)
        self.dataset_name = dataset_name
        self.dataloaders = {}
        self.description = data['description']
        for key, df in data.items():
            if key == 'description':
                continue
            if 'metaE' not in key:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.description, device), batch_size=bsz, shuffle=shuffle)
            else:
                self.dataloaders[key] = DataLoader(Movielens1MbaseDataset(dataset_name, df, self.description, device), batch_size=bsz, shuffle=False)
        self.keys = list(self.dataloaders.keys())
        self.item_features = ['year', 'title', 'genres', 'count']
                                

    def __getitem__(self, name):
        assert name in self.keys, '{} not in keys of datasets'.format(name)
        return self.dataloaders[name]
        
        
        

        

