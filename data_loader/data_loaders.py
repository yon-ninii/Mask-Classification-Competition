from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from base import BaseDataLoader
from .dataset import MaskDataset, MaskGlobDataset
from glob import glob
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose, CoarseDropout, CLAHE, CenterCrop, RandomCrop, OpticalDistortion, GridDistortion, Perspective, Crop, Resize

class MaskDataLoader(BaseDataLoader):
    """
    Competition Mask DataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=False, dataset='default'):
        # default transform
        trsfm = Compose([
            Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
            Resize(224, 224),
            ToTensorV2()
        ])

        self.data_dir = data_dir
        self.dataset = self._get_dataset(dataset, data_dir, trsfm, training)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def _get_dataset(self, dataset, data_dir, trsfm, train):
        return MaskDataset(data_dir, trsfm, train)



class MaskSplitLoader(DataLoader):
    """
    Competition Mask DataLoader. Split by profile
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, dataset='default'):
        # default transform

        self.data_dir = data_dir = Path(data_dir)
        self.df = pd.read_csv(data_dir / 'train.csv')
        self.paths = self.df['path'].to_numpy()
        self.labels = (self.df['gender'].map({'male': 0, 'female': 1}) * 3 + self.df['age'] // 30).to_numpy()
        train_transforms, valid_transforms = get_transform('clahe_od')
        
        if validation_split == 0:
            train_idx = range(len(self.paths))
            valid_idx = []
        else:
            s = StratifiedShuffleSplit(n_splits=1, test_size=validation_split, random_state=0)
            s.get_n_splits()
            train_idx, valid_idx = next(s.split(self.paths, self.labels))
        
        data_dir_str = str(data_dir)
        
        # new_path1 = data_dir_str.replace('train', '60HFlip')
        # new_path2 = data_dir_str.replace('train', '60Rotaten45')
        # new_path3 = data_dir_str.replace('train', '60Rotatep45')
        new_path4 = data_dir_str.replace('train', '60HFRotatep45')
        new_path5 = data_dir_str.replace('train', '60HFRotaten45')

        # ds1 = MaskGlobDataset(new_path1, train_transforms, paths=self.paths[train_idx])
        # ds2 = MaskGlobDataset(new_path2, train_transforms, paths=self.paths[train_idx])
        # ds3 = MaskGlobDataset(new_path3, train_transforms, paths=self.paths[train_idx])
        ds4 = MaskGlobDataset(new_path4, train_transforms, paths=self.paths[train_idx])
        ds5 = MaskGlobDataset(new_path5, train_transforms, paths=self.paths[train_idx])
                
        trainset = MaskGlobDataset(data_dir, train_transforms, paths=self.paths[train_idx])
        
        self.trainset = ConcatDataset([ds4, ds5, trainset]) #ds1, ds2, ds3, 
        
        self.validset = MaskGlobDataset(data_dir, valid_transforms, paths=self.paths[valid_idx])
        self.n_samples = len(self.trainset)

        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }
        super().__init__(self.trainset, **self.init_kwargs)
    
    def _split_sampler(self):
        raise Exception('do not use _split_sampler')
    
    def split_validation(self):
        dl = DataLoader(self.validset, **self.init_kwargs)
        dl.n_samples = len(self.validset)
        return dl
    
#    CLAHE, CenterCrop, GridDistortion, Perspective, RandomCrop, ToTensorV2

    
def get_transform(t_name):
    Cutout = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
        Resize(224, 224),
        ToTensorV2(),
    ])
    CLAHE_OD = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        CLAHE(clip_limit=(5.0, 15.0), tile_grid_size=(8, 8), always_apply=False, p=0.3),
        OpticalDistortion(distort_limit=0.65, shift_limit=0.15, interpolation=1, border_mode=4, p=0.3),
        Resize(224, 224),
        ToTensorV2()
    ])
    RC = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        RandomCrop(height=350, width=200, always_apply=False, p=0.5),
        Resize(224, 224),
        ToTensorV2()
    ])
    CLAHE_CC = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        CLAHE(clip_limit=(5.0, 15.0), tile_grid_size=(8, 8), always_apply=False, p=0.5),
        CenterCrop(height=300, width=160, always_apply=False, p=0.5),
        Resize(224, 224),
        ToTensorV2()
    ])
    GD_Perspect = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, p=0.5),
        Perspective (scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, p=0.5),
        Resize(224, 224),
        ToTensorV2()
    ])
    Normal_train = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        Resize(224, 224),
        ToTensorV2()
    ])
    valid_transform = Compose([
        Crop(p=1, x_min=0, y_min=50, x_max=384, y_max=512),
        Resize(224, 224),
        ToTensorV2()
    ])
    TF_names = {'cutout':Cutout, 'clahe_od':CLAHE_OD, 'rc':RC, 'clahe_cc':CLAHE_CC, 'gd_perspect':GD_Perspect, 'normal':Normal_train}
    return TF_names[t_name], valid_transform