import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import albumentations as A
from grid import create_grid
import sys

# Augmentation
def augmentation(aug: str):
    if aug == 'd4':
        return A.Compose([
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
        ], is_check_shapes=False) # is_check_shapes my_
    elif aug == 'rotation':
        return A.Compose([
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=30, scale_limit=0.2, p=0.75)
        ], is_check_shapes=False) # is_check_shapes my_
    else:
        raise ValueError


# Ash-color false color
def rescale_range(x, f_min, f_max):
    # Rescale [f_min, f_max] to [0, 1]
    return (x - f_min) / (f_max - f_min)


def ash_color(x):
    """
    False color for contrail annotation
    x (array): (3, H, W) -> (3, H, W)
    """
    r = rescale_range(x[2] - x[1], -4, 2)  # ch 7 - 6
    g = rescale_range(x[1] - x[0], -4, 5)  # ch 6 - 3
    b = rescale_range(x[1], 243, 303)      # ch 6

    x = torch.stack([r, g, b], axis=0)
    x = 1 - x

    return x


class Dataset(torch.utils.data.Dataset):
    """
    d = dataset[i]
    x: image (C, H, W)
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
        
        # Antialiasing is a technique to reduce the visual artifacts known as "aliasing" that can occur when resizing or rendering images. These artifacts often manifest as jagged or stair-stepped edges, especially when an image is resized to a smaller size or displayed at a lower resolution
        self.resize = nn.Identity() if nc == 256 else T.Resize(nc, antialias=False)

        assert nc == 1024
        
        nc_grid = 512
        self.grid = create_grid(nc_grid, offset=0.5)  # y_sym is 512 x 512 for unet1024

        self.y_sym_mode = cfg['data']['y_sym_mode']
        assert self.y_sym_mode in ['bilinear', 'nearest']

        self.augment_prob = cfg['data']['augment_prob']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        filename = r['filename']

        # Load
        with h5py.File(filename, 'r') as f:
            # f['x'].shape => (4, 3, 256, 256)
            
            # h5 file contains t=1,2,3,4, and index 3 is t=4
            x = f['x'][3, :]  # (3, H, W)
            # x.shape => (3, 256, 256)

            # Target
            if 'annotation_mean' in f:
                if self.annotation_mean == 'mix':
                    y = 0.5 * (f['annotation_mean'][:].astype(np.float32) + f['y'][:].astype(np.float32))
                elif self.annotation_mean is True:
                    # f['annotation_mean'].shape => (1, 256, 256)
                    y = f['annotation_mean'][:].astype(np.float32)
                else:
                    y = f['y'][:].astype(np.float32)      # (1, H, W)
            else:
                y = None

            # Ground truth for validation
            # f['y'].shape => (1, 256, 256)
            label = torch.from_numpy(f['y'][:].astype(np.float32))
            

        x = torch.from_numpy(x)
        if y is not None:
            y = torch.from_numpy(y)

        # Create color image
        x = ash_color(x)
        # x.shape => torch.Size([3, 256, 256])
        x = self.resize(x)
        # x.shape => torch.Size([3, 1024, 1024])

        if y is not None:
            # y.unsqueeze(0).shape  => torch.Size([1, 1, 256, 256])               (N/B, C, H, W) 
            # self.grid.shape => torch.Size([1, 512, 512, 2])                     (N/B, H, W, 2)            
            # self.y_sym_mode => bilinear
            # y.unique() => tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
            
            # grid-based sampling of input data. 
            # 2D coordinates (x, y) represent the sampling points on the input grid.
            # for each output location output[1, C/1, 512, 512], the size-2 vector grid[1, 512, 512] specifies input pixel locations x and y, which are used to interpolate the output value output[1, C/1, 512, 512].
            # self.grid => values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.
            y_sym = F.grid_sample(y.unsqueeze(0), self.grid,
                                  mode=self.y_sym_mode, padding_mode='border',
                                  align_corners=False)
            # y_sym.unique() =>
            # tensor([0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781, 0.0938, 0.1094, 0.1250, ..., 0.9531, 0.9688, 0.9844, 1.0000])           

            # y_sym.shape => torch.Size([1, 1, 512, 512])
            y_sym = y_sym.squeeze(0)
            # y_sym.shape => torch.Size([1, 512, 512])
            

        # Augment
        w_original = 1.0
        if self.augment is not None and np.random.random() < self.augment_prob:
            w_original = 0.0

            x = x.permute(1, 2, 0).numpy()  # => (H, W, C)
            y_sym = y_sym.permute(1, 2, 0).numpy()
            
            # x.shape, y_sym.shape => (1024, 1024, 3), (512, 512, 1)

            aug = self.augment(image=x, mask=y_sym)

            x = torch.from_numpy(aug['image'].transpose(2, 0, 1))     # => (C, H, W)
            y_sym = torch.from_numpy(aug['mask'].transpose(2, 0, 1))  # array (1, 256, 256)

        d = {'x': x, 'w': np.float32(w_original)}  # 1 if y is original not augmented

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
                                           num_workers= num_workers, #num_workers, # 1 my_
                                           shuffle=shuffle,
                                           drop_last=drop_last)
