import torch
import numpy as np

from medpy import io
from pathlib import Path
from itertools import product, chain
from collections import OrderedDict
from torch.utils.data import Dataset, Sampler

class aortaIMhaDataset(Dataset):
    """Dataset for loading CT images from mha format
    """
    def __init__(self, datapath, names, *, channels=None):
        """
            :NOTE:
                format of channels:

                { # yaml notation
                    CHANNEL_NAME_1:
                        MIN_HU: VALUE
                        MAX_HU: VALUE
                    CHANNEL_NAME_2:
                        ...
                }

            :args:
                datapath: pathlib.Path
                    directiry path with data
                names: list of str
                    names of data to load
                channels: dict, see NOTE
                    option to online form channels for imgs
        """
        self.channels = channels

        self.imgs = OrderedDict()

        self.keys = list()
        self.shapes = list()

        for name in names:
            image, _ = io.load(datapath / 'images' / f'subject{name:03}_CTA.mha')

            self.imgs[name] = image

            self.keys.append(name)
            self.shapes.append(image.shape)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        key, selector = idx

        voxel = self.imgs[key][selector]
        voxel = voxel[:, :, :, None]
        
        if self.channels:
            voxel = self.split_images(voxel, self.channels)

        voxel = np.moveaxis(voxel, -1, 0)

        return voxel, key, selector

    @staticmethod
    def normalize_HU(images, *, MIN_HU, MAX_HU):
        images = (images - MIN_HU) / (MAX_HU - MIN_HU)
        return images.clip(min=0., max=1.)
    
    @staticmethod
    def split_images(images, channels):
        image_channels = ()

        for options in channels.values():
            image_channel = aortaIMhaDataset.normalize_HU(images, MIN_HU=options['MIN_HU'], MAX_HU=options['MAX_HU'])
            image_channels = (*image_channels, image_channel)

        return np.concatenate(image_channels, axis=-1)

    @staticmethod
    def collate_fn(batch):
        voxels, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = torch.tensor(voxels, dtype=torch.float)

        return voxels, selections

class aortaIMMhaDataset(aortaIMhaDataset):
    """Dataset for loading CT images and segmentation masks from mha format
    """
    def __init__(self, datapath, names, *, augs=None, channels=None):
        super().__init__(datapath, names, channels=None)

        self._channels = channels
        
        self.masks = dict()

        for idx, name in enumerate(names):
            mask, _ = io.load(datapath / 'masks' / f'subject{name:03}_label.mha')

            self.masks[name] = mask

        for idx, name in enumerate(names):
            mask = self.masks[name]
            
            xsize, ysize, zsize = mask.shape
            hhxsize, hhysize = xsize // 4, ysize // 4
            
            zids = np.where(mask.sum((0, 1)) > 0)

            zmin = max(np.min(zids)-20, 0)
            zmax = min(np.max(zids)+20, zsize-1)

            self.imgs[name] = self.imgs[name][hhxsize:-hhxsize, hhysize:-hhysize, zmin:zmax+1]
            self.masks[name] = self.masks[name][hhxsize:-hhxsize, hhysize:-hhysize, zmin:zmax+1]

            self.shapes[idx] = self.masks[name].shape

        self.dists = dict()
        
        for idx, name in enumerate(names):
            dist = np.zeros((24, self.masks[name].shape[2]), dtype=np.float32) + 1
            
            masks = np.moveaxis(self.masks[name], 2, 0)
            
            for idx, mask in enumerate(masks):
                unique, counts = np.unique(mask, return_counts=True)
            
                for u, c in zip(unique, counts):
                    dist[u, idx] = c
            
            dist = np.apply_along_axis(
                lambda arr: np.convolve(
                    arr,
                    np.ones(60) / 60,
                    'valid'
                ),
                1,
                dist
            )
            
            self.dists[name] = dist / dist.sum(axis=1, keepdims=True)

        self.augs = augs

    def __getitem__(self, idx):
        voxel, key, selector = aortaIMhaDataset.__getitem__(self, idx)
        mask = self.masks[key][selector]

        voxel = np.moveaxis(voxel, 0, -1)
        
        if self.augs:
            auged = self.augs(voxel=voxel, mask=mask)
            voxel = auged['voxel']
            mask = auged['mask']

        if self._channels:
            voxel = self.split_images(voxel, self._channels)
        
        voxel = np.moveaxis(voxel, -1, 0)

        return voxel, mask, key, selector

    @staticmethod
    def collate_fn(batch):
        voxels, masks, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = np.array(voxels, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8)

        voxels = torch.tensor(voxels, dtype=torch.float)
        masks = torch.tensor(masks, dtype=torch.long)

        return voxels, masks, selections

    @staticmethod
    def collate_fn_eval(batch):
        voxels, masks, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = np.array(voxels, dtype=np.float32)
        voxels = torch.tensor(voxels, dtype=torch.float)

        return voxels, selections

class aortaIMMhaDataset2(aortaIMhaDataset):
    """Dataset for loading CT images and segmentation masks from mha format
    """
    def __init__(self, datapath, names, *, augs=None, channels=None):
        super().__init__(datapath, names, channels=None)

        self._channels = channels
        
        self.masks = dict()

        for idx, name in enumerate(names):
            mask, _ = io.load(datapath / 'masks' / f'subject{name:03}_label.mha')

            self.masks[name] = mask

        for idx, name in enumerate(names):
            mask = self.masks[name]
            
            xsize, ysize, zsize = mask.shape
            hhxsize, hhysize = xsize // 4, ysize // 4

            self.imgs[name] = self.imgs[name][hhxsize:-hhxsize, hhysize:-hhysize]
            self.masks[name] = self.masks[name][hhxsize:-hhxsize, hhysize:-hhysize]

            self.shapes[idx] = self.masks[name].shape

        self.augs = augs

    def __getitem__(self, idx):
        voxel, key, selector = aortaIMhaDataset.__getitem__(self, idx)
        mask = self.masks[key][selector]

        voxel = np.moveaxis(voxel, 0, -1)
        
        if self.augs:
            auged = self.augs(voxel=voxel, mask=mask)
            voxel = auged['voxel']
            mask = auged['mask']

        if self._channels:
            voxel = self.split_images(voxel, self._channels)
        
        voxel = np.moveaxis(voxel, -1, 0)

        return voxel, mask, key, selector

    @staticmethod
    def collate_fn(batch):
        voxels, masks, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = np.array(voxels, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8)

        voxels = torch.tensor(voxels, dtype=torch.float)
        masks = torch.tensor(masks, dtype=torch.long)

        return voxels, masks, selections

    @staticmethod
    def collate_fn_eval(batch):
        voxels, masks, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = np.array(voxels, dtype=np.float32)
        voxels = torch.tensor(voxels, dtype=torch.float)

        return voxels, selections

class aortaIMMThaDataset(aortaIMhaDataset):
    """Dataset for loading CT images and segmentation masks from mha format
    """
    def __init__(self, datapath, names, *, channels=None):
        super().__init__(datapath, names, channels=None)

        self._channels = channels
        
        self.masks = dict()

        for idx, name in enumerate(names):
            mask, _ = io.load(datapath / 'masks' / f'subject{name:03}_label.mha')

            self.masks[name] = mask

    def __getitem__(self, idx):
        voxel, key, selector = aortaIMhaDataset.__getitem__(self, idx)
        mask = self.masks[key][selector]

        voxel = np.moveaxis(voxel, 0, -1)

        if self._channels:
            voxel = self.split_images(voxel, self._channels)
        
        voxel = np.moveaxis(voxel, -1, 0)

        return voxel, mask, key, selector

    @staticmethod
    def collate_fn_eval(batch):
        voxels, masks, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = np.array(voxels, dtype=np.float32)
        voxels = torch.tensor(voxels, dtype=torch.float)

        return voxels, selections

def voxel_random_selector(voxel_shape, case_keys, shapes, n_count=1):
    assert type(case_keys) in {list, tuple}
    assert len(case_keys) == len(shapes)

    n_cases = np.arange(len(case_keys))

    for _ in np.arange(n_count):
        case_idx = np.random.choice(n_cases)

        case_key = case_keys[case_idx]
        case_shape = shapes[case_idx]

        selector = ()

        for voxel_size, case_size in zip(voxel_shape, case_shape):
            if voxel_size is None:
                selector = (*selector, slice(0, None))
            else:
                assert voxel_size <= case_size

                if voxel_size == case_size:
                    coord = 0
                else:
                    coord = np.random.choice(case_size - (voxel_size - 1))

                selector = (*selector, slice(coord, coord + voxel_size))

        yield case_key, selector

class VoxelRandomSampler(Sampler):
    def __init__(self, voxel_shape, case_keys, shapes, n_count=1):
        self.voxel_shape = voxel_shape
        self.case_keys = case_keys
        self.shapes = shapes
        self.n_count = n_count

    def __iter__(self):
        yield from voxel_random_selector( self.voxel_shape,
                                          self.case_keys,
                                          self.shapes,
                                          self.n_count )

    def __len__(self):
        return self.n_count

def voxel_weighted_selector(dataset, voxel_shape, case_keys, shapes, n_count=1):
    assert type(case_keys) in {list, tuple}
    assert len(case_keys) == len(shapes)

    n_cases = np.arange(len(case_keys))
    p = np.array([0.00167224, 0.03846154, 0.02341137, 0.05183946, 0.02675585,
       0.03010033, 0.01505017, 0.00167224, 0.00167224, 0.02675585,
       0.10869565, 0.01170569, 0.12207358, 0.05351171, 0.07525084,
       0.07525084, 0.00501672, 0.02842809, 0.03511706, 0.13043478,
       0.09197324, 0.02508361, 0.02006689])

    classes = np.random.choice(np.arange(1, 24), size=n_count)

    for cdx in np.arange(n_count):
        case_idx = np.random.choice(n_cases)

        case_key = case_keys[case_idx]
        case_shape = shapes[case_idx]

        selector = ()

        for idx, (voxel_size, case_size) in enumerate(zip(voxel_shape, case_shape)):
            if voxel_size is None:
                selector = (*selector, slice(0, None))
            else:
                assert voxel_size <= case_size

                if voxel_size == case_size:
                    coord = 0
                else:
                    if idx == 2:
                        class_ = classes[cdx]

                        coord = np.random.choice(case_size - (voxel_size - 1), p=dataset.dists[case_key][class_])
                    else:
                        coord = np.random.choice(case_size - (voxel_size - 1))

                selector = (*selector, slice(coord, coord + voxel_size))

        yield case_key, selector

class VoxelWeightedSampler(Sampler):
    def __init__(self, dataset, voxel_shape, case_keys, shapes, n_count=1):
        self.dataset = dataset
        self.voxel_shape = voxel_shape
        self.case_keys = case_keys
        self.shapes = shapes
        self.n_count = n_count

    def __iter__(self):
        yield from voxel_weighted_selector( self.dataset,
                                            self.voxel_shape,
                                            self.case_keys,
                                            self.shapes,
                                            self.n_count )

    def __len__(self):
        return self.n_count

def voxel_sequential_selector(voxel_shape, case_keys, shapes, steps):
    assert type(case_keys) in {list, tuple}
    assert len(voxel_shape) == len(steps)

    for idx, case_key in enumerate(case_keys):
        ranges = ()

        for step, voxel_size, case_size in zip(steps, voxel_shape, shapes[idx]):
            assert case_size >= voxel_size

            range_ = chain(np.arange(0, case_size - voxel_size, step), (case_size - voxel_size ,))
            ranges = (*ranges, range_)

        for point in product(*ranges):
            selector = ()

            for coord, voxel_size in zip(point, voxel_shape):
                selector = (*selector, slice(coord, coord+voxel_size))

            yield case_key, selector

class VoxelSequentialSampler(Sampler):
    def __init__(self, voxel_shape, case_keys, shapes, steps):
        self.voxel_shape = voxel_shape
        self.case_keys = case_keys
        self.shapes = shapes
        self.steps = steps

    def __iter__(self):
        yield from voxel_sequential_selector( self.voxel_shape,
                                              self.case_keys,
                                              self.shapes,
                                              self.steps )

    def __len__(self):
        return len([*voxel_sequential_selector( self.voxel_shape,
                                                self.case_keys,
                                                self.shapes,
                                                self.steps )])
