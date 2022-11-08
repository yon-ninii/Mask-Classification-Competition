from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import os
import glob
from PIL import Image
import numpy as np

filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
masklabels = [1, 0, 0, 0, 0, 0, 2]
aug_masklabels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]

class MaskDataset(Dataset):
    def __init__(self, root, transform, training=False):
        self.root = root
        self.is_train = training
        self.transform = transform
        
        self.paths = []
        self.df = pd.read_csv(os.path.join(root, 'eval', 'info.csv'))
        self.paths = [os.path.join(root, 'eval', 'images', img_id) for img_id in self.df.ImageID]
                    

    def __getitem__(self, index):
        image = np.array(Image.open(self.paths[index]))
        if self.transform:
            image = self.transform(image=image)['image']
        return image
        
    def __len__(self):
        return len(self.paths)