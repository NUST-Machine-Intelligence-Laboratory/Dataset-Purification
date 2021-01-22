import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pickle

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class dataset_from_list(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None, cached=False):
        print('Dataset from list')
        self.transform = transform
        self.samples = []
        self.cached = cached
        self.loader = pil_loader

        if isinstance(data_list,str):
            data_list=[data_list]
        for dl in data_list:
            print('Data list:   ', dl)
            if dl[-3:] == 'txt':
                data = np.genfromtxt(dl, dtype=str)
                assert data.shape[0] > 0, 'data list is wrong, no images'
                for idx in range(data.shape[0]):
                    item = (data[idx, 0].replace('zcy','zhangchuanyi').replace('fg-web-data','web-data'), int(data[idx, 1]))
                    self.samples.append(item)
            elif dl[-3:] == 'pkl':
                f = open(dl, 'rb')
                data = pickle.load(f)
                f.close()
                assert len(data) > 0, 'data list is wrong, no images'
                for idx in range(len(data)):
                    d = data[idx]
                    item = (d[0].replace('zcy','zhangchuanyi').replace('fg-web-data','web-data'), int(d[1]))
                    self.samples.append(item)
            else:
                raise ValueError("Wrong data list")

        if self.cached:
            print('load all images cached')
            self.images=[]
            for sample in self.samples:
                path, target = sample
                # image = self.loader(path)
                self.images.append(self.loader(path))

        print('image number: ', len(self.samples))

    def __getitem__(self, index):
        path, target = self.samples[index]

        if self.cached == False:
            # print(path)
            sample = self.loader(path)
        else:
            sample = self.images[index]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, index, path

    def __len__(self):
        """
        Length of the dataset

        Return:
            [int] Length of the dataset
        """
        return len(self.samples)