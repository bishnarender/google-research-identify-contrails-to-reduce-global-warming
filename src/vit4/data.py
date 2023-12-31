import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import albumentations as A
import sys
from grid import create_grid

di = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'


# Augmentation
def augmentation(aug: str):
    if aug == 'd4':
        return A.Compose([
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
        ])
    elif aug == 'rotation':
        return A.Compose([
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=30, scale_limit=0.2, p=0.75)
        ])
    else:
        raise ValueError


# Ash false color
def rescale_range(x, f_min, f_max):
    # Rescale [f_min, f_max] to [0, 1]
    return (x - f_min) / (f_max - f_min)


def ash_color(x):
    """
    False color for contrail annotation
    x (array): (T, C, H, W) -> (T, 3, H, W)
    """
    r = rescale_range(x[:, 2] - x[:, 1], -4, 2)
    g = rescale_range(x[:, 1] - x[:, 0], -4, 5)
    b = rescale_range(x[:, 1], 243, 303)

    x = torch.stack([r, g, b], axis=1)
    x = 1 - x

    return x


class Dataset(torch.utils.data.Dataset):
    """
    d = dataset[i]
    x: image (3, 512, 512) 4 timesteps in one image
    y: segmentation mask (1, H, W)
    """
    def __init__(self, df, cfg, *, augment=False):
        self.df = df
        self.augment = None
        if augment and cfg['data']['augment']:
            self.augment = augmentation(cfg['data']['augment'])

        self.annotation_mean = cfg['data']['annotation_mean']
        assert self.annotation_mean in ['mix', True, False]

        nc = cfg['data']['resize']
        self.resize = nn.Identity() if nc == 256 else T.Resize(nc, antialias=False)

        self.grid = create_grid(nc, offset=0.5)

        self.y_sym_mode = cfg['data']['y_sym_mode']
        assert self.y_sym_mode in ['bilinear', 'nearest']

        self.augment_prob = cfg['data']['augment_prob']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        filename = r['filename']

        # Load
        T = 4
        with h5py.File(filename, 'r') as f:
            x = f['x'][:]  # (T=4, C=3, H, W)
            # x.shape => (4, 3, 256, 256)
            
            # Target
            if 'annotation_mean' in f:
                if self.annotation_mean == 'mix':
                    y = 0.5 * (f['annotation_mean'][:].astype(np.float32) + f['y'][:].astype(np.float32))
                elif self.annotation_mean is True:
                    y = f['annotation_mean'][:].astype(np.float32)
                else:
                    y = f['y'][:].astype(np.float32)    # (1, H, W)
            else:
                y = None

            # Ground truth for validation
            label = torch.from_numpy(f['y'][:].astype(np.float32))

        x = torch.from_numpy(x)
        if y is not None:
            y = torch.from_numpy(y)

        # Create color image
        x = ash_color(x)
        # x.shape => torch.Size([4, 3, 256, 256])
        x = self.resize(x)
        # x.shape => torch.Size([4, 3, 512, 512])    

        if y is not None:

            # y.shape, self.grid.shape => torch.Size([1, 256, 256]), torch.Size([1, 512, 512, 2])
            y_sym = F.grid_sample(y.unsqueeze(0), self.grid,
                                  mode=self.y_sym_mode, padding_mode='border',
                                  align_corners=False).squeeze(0)
            # y_sym.shape => torch.Size([1, 512, 512])            

        # Augment
        w_original = 1.0
        _, _, h, w = x.shape
        # x.shape => torch.Size([4, 3, 512, 512])
        if self.augment is not None and np.random.random() < self.augment_prob:
            w_original = 0.0
            x = x.reshape(T * 3, h, w)
            x = x.permute(1, 2, 0).numpy()  # (T * C, H, W) -> (H, W, T * C)
            y_sym = y_sym.permute(1, 2, 0).numpy()
            

            aug = self.augment(image=x, mask=y_sym)

            x = torch.from_numpy(aug['image'].transpose(2, 0, 1))     # (T, H, W, C) -> (T, C, H, W)
            y_sym = torch.from_numpy(aug['mask'].transpose(2, 0, 1))  # array (1, 256, 256)

            x = x.reshape(T, 3, h, w)
            # x.shape => torch.Size([4, 3, 512, 512])
        

        # Concatenate 4 images
        nc = x.size(2)  # (T, 3, H, W)
        assert nc in [256, 512]
        nc2 = 2 * nc
        x = x.numpy()
        x4 = np.zeros((3, nc2, nc2), dtype=np.float32)
        # x4.shape => (3, 1024, 1024)

        x4[:, :nc, :nc] = x[3]  #- t=4  #- x4[:, :512, :512] #- top-left section of 1024 x 1024 
        x4[:, :nc, nc:] = x[0]  # x4[:, :512, 512:] #- top-right section of 1024 x 1024 
        x4[:, nc:, :nc] = x[1]  # x4[:, 512:, :512] #- bottom-left section of 1024 x 1024 
        x4[:, nc:, nc:] = x[2]  # x4[:, 512:, 512:] #- bottom-right section of 1024 x 1024 
        
        # x4.shape => (3, 1024, 1024)
        
        # Return values
        d = {'x': torch.from_numpy(x4), 'w': np.float32(w_original)}  # 1 if y is original not augmented

        if y is not None:
            d['y_sym'] = y_sym
            d['y'] = y

        if label is not None:
            d['label'] = label
        return d


class Data:
    def __init__(self, data_type, data_dir, *, debug=False):
        # Load filename list
        df = pd.read_csv('%s/%s.csv' % (data_dir, data_type))
        if debug:
            df = df.iloc[:100]

        self.df = df

    def __len__(self):
        return len(self.df)

    def dataset(self, idx, cfg, augment, *, positive_only=False):
        df = self.df.iloc[idx] if idx is not None else self.df

        if positive_only:
            df = df[df.label_sum > 0]
            print('Data positive only: %d' % len(df))

        return Dataset(df, cfg, augment=augment)

    def loader(self, idx, cfg, *, augment=False, shuffle=False, drop_last=False):
        batch_size = cfg['train']['batch_size']
        num_workers = cfg['train']['num_workers']
        positive_only = cfg['data']['positive_only']

        ds = self.dataset(idx, cfg, augment, positive_only=positive_only)
        return torch.utils.data.DataLoader(ds,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=shuffle,
                                           drop_last=drop_last)
