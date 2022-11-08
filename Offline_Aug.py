import pandas as pd
import os
import glob
import albumentations as A
import cv2

root = '/opt/ml/input/data'

filenames = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']

paths = []
df = pd.read_csv(os.path.join(root, 'train', 'train.csv'))
for path in df['path']:
    p = [os.path.join(root, 'train', 'images', path, f+'.jpg') for f in filenames]
    paths.extend(p)

Horizontal = A.Compose([
    A.HorizontalFlip(p=1)
])

Blur = A.Compose([
    A.MedianBlur(blur_limit=5, p=1)
])

Rotation = A.Compose([
    A.Rotate(limit=30, p=1)
])

#for p in paths:
for i in range(2):
    p = paths[i]
    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _Horizontal = Horizontal(image=image)
    _Blur = Blur(image=image)
    _Rotation = Rotation(image=image)
    
    cv2.imshow('horizontal', _Horizontal["image"])
    cv2.imshow('blur', _Blur["image"])
    cv2.imshow('rotate', _Rotation["image"])

    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    


'''
mytransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(244),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(0, 360)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
])

myvaltransform =transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
'''