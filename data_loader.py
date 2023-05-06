import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.torch_form()

    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == 'train':
            if self.args.pretrained_path:
                # finetune
                self.X = np.load(f"./data/S{s:02}_train_X.npy")
                self.y = np.load(f"./data/S{s:02}_train_y.npy")
            else:
                # pretrain
                self.X = combine_npy(s, '_X', 'train')
                self.y = combine_npy(s, '_y', 'train')
        else:
            if self.args.pretrained_path:
                # finetune
                self.X = np.load(f"./data/S{s:02}_test_X.npy")
                self.y = np.load(f"./answer/S{s:02}_y_test.npy")
            else:
                # pretrain
                self.X = combine_npy(s, '_X', 'val')
                self.y = combine_npy(s, '_y', 'val')

        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample

def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"
    trainset = CustomDataset(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader

def combine_npy(s, t, h): # s = 1-9, t = '_y', '_X', h = 'train', 'val', 'test'

    # get npy files
    npy_files = os.listdir('./data/')
    npy_files.sort()

    # combine npy files by subjects
    npy_list = []
    for npy_file in npy_files:
        if h == 'train':
            if t in npy_file:
                if str(s) not in npy_file:
                    y = np.load('./data/'+npy_file)
                    # concatenate
                    if len(npy_list) == 0:
                        npy_list = y
                    else:
                        npy_list = np.concatenate((npy_list, y), axis=0)
        else:
            if t in npy_file:
                if str(s) in npy_file:
                    y = np.load('./data/'+npy_file)
                    # concatenate
                    if len(npy_list) == 0:
                        npy_list = y
                    else:
                        npy_list = np.concatenate((npy_list, y), axis=0)
    return npy_list
