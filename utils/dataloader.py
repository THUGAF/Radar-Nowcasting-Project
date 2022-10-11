from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import numpy as np


class MyDataset(Dataset):
    def __init__(self, start, end, root='data/', is_sample=False):
        self.range = range(start, end)
        self.psuedo_range = range(start + 1, end)
        self.is_sample = is_sample
        self.root = root
        
        # 5D inputs shape: S*B*C*H*W
        self.data = self._load_ith_npz(start)
            
        for i in self.psuedo_range:
            self.data = torch.cat((self.data, self._load_ith_npz(i)), 1)

        # Normalization
        self.data /= 180.0  # max value of the samples
        self.data[self.data > 1.0] = 1.0

    def __getitem__(self, index):
        return self.data[:, index]

    def __len__(self):
        if self.is_sample:
            return len(self.range)
        else:
            return 100 * len(self.range)
    
    def _load_ith_npz(self, i):
        return torch.from_numpy(np.load(self.root + str(i) + '.npz')['arr_0']).float()


def load_data(data_range, batch_size):
    print('loading data ...')
    dataset = MyDataset(data_range[0], data_range[1])
    train_set, val_set, test_set = random_split(dataset, \
        [round(10 / 14 * len(dataset)), round(2 / 14 * len(dataset)), round(2 / 14 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for i, data in enumerate(train_loader):
        data = data.transpose(1, 0)
        
        if i == 0:
            data_size = data.size()
            print('\n[Train Loader]')
            print(' batch num:', len(train_loader))
            print(' data size:', data_size)
    
    for i, data in enumerate(val_loader):
        data = data.transpose(1, 0)
            
        if i == 0:
            data_size = data.size()
            print('\n[Val Loader]')
            print(' batch num:', len(val_loader))
            print(' data size:', data_size)

    for i, data in enumerate(test_loader):
        data = data.transpose(1, 0)
        
        if i == 0:
            data_size = data.size()
            print('\n[Test Loader]')
            print(' batch num:', len(test_loader))
            print(' data size:', data_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data([0, 7], batch_size=10)
    
