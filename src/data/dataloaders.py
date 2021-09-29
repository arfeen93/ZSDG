import numpy as np
from random import random
from PIL import Image, ImageOps

import torch
import torch.utils.data as data
import torchvision


class BaselineDataset(data.Dataset):
    def __init__(self, fls, transforms=None):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        # if sample_domain=='sketch' or sample_domain=='quickdraw':
        #     sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        # else:
        #     sample = Image.open(self.fls[item]).convert(mode='RGB')
        sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss

    def __len__(self):
        return len(self.fls)


class CuMixloader(data.Dataset):
    
    def __init__(self, fls, clss, doms, dict_domain, transforms=None):
        
        self.fls = fls
        self.clss = clss
        self.domains = doms
        self.dict_domain = dict_domain
        self.transforms = transforms

    def __getitem__(self, item):

        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        
        clss = self.clss[item]

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample, clss, self.dict_domain[sample_domain]

    def __len__(self):
        return len(self.fls)


class JigsawDataset(data.Dataset):
    def __init__(self, fls, transforms=None, jig_classes=30, bias_whole_image=0.9):
        
        self.fls = fls
        self.clss = np.array([f.split('/')[-2] for f in fls])
        self.domains = np.array([f.split('/')[-3] for f in fls])

        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.bias_whole_image = bias_whole_image # biases the training procedure to show the whole image more often
    
        self._image_transformer = transforms['image']
        self._augment_tile = transforms['tile']
        
        def make_grid(x):
            return torchvision.utils.make_grid(x, self.grid_size, padding=0)
        self.returnFunc = make_grid

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def get_image(self, item):
        
        sample_domain = self.domains[item]
        if sample_domain=='sketch' or sample_domain=='quickdraw':
            sample = ImageOps.invert(Image.open(self.fls[item])).convert(mode='RGB')
        else:
            sample = Image.open(self.fls[item]).convert(mode='RGB')
        return self._image_transformer(sample)
        
    def __getitem__(self, item):
        
        img = self.get_image(item)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1) # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            
        data = torch.stack(data, 0)
        data = self.returnFunc(data)
        
        return torch.cat([self._augment_tile(img), data], 0), order, self.clss[item]

    def __len__(self):
        return len(self.fls)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('/home/soumava/TTT-ZSDG/src/data/permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm