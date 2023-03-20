import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser



#---- Notice-----#
# In order to use this dataloader, please run:
# 1. data-preprocess.ipynb in ./notebooks, for a better organized csv file
# 2. preprocess.py in ./data_preprocess, for images in the same and smaller size



# disease_labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
DISEASE_LABELS = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

class ISICDataset(Dataset):
    def __init__(self, img_data_dir, df_data, image_size, augmentation=False, pseudo_rgb = False):
        self.df_data = df_data
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        # self.labels = ['melanoma','nevus','basal cell carcinoma','actinic keratosis','benign keratosis','dermatofibroma',
        #   'vascular lesion','squamous cell carcinoma','others']
        self.labels=DISEASE_LABELS

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        ])

        self.samples = []
        # for idx, _ in enumerate(tqdm(range(len(self.df_data)), desc='Loading Data')):
        for idx in tqdm((self.df_data.index), desc='Loading Data'):
            img_path = img_data_dir + self.df_data.loc[idx, 'path_preproc']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.df_data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        image = torch.from_numpy(sample['image']).unsqueeze(0)
        label = torch.from_numpy(sample['label'])

        image = image.squeeze(dim=0)
        image = torch.permute(image, dims=(2, 0, 1))

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)



        return {'image': image, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        image = imread(sample['image_path']).astype(np.float32)

        return {'image': image, 'label': sample['label']}



class ISICDataModule(pl.LightningDataModule):
    def __init__(self, img_data_dir,csv_file_img, image_size, pseudo_rgb, batch_size, num_workers,augmentation):
        super().__init__()
        self.img_data_dir = img_data_dir
        self.csv_file_img = csv_file_img

        df_train,df_valid,df_test = self.dataset_split(self.csv_file_img)
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation=augmentation


        self.train_set = ISICDataset(self.img_data_dir,self.df_train, self.image_size, augmentation=augmentation, pseudo_rgb=pseudo_rgb)
        self.val_set = ISICDataset(self.img_data_dir,self.df_valid, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)
        self.test_set = ISICDataset(self.img_data_dir,self.df_test, self.image_size, augmentation=False, pseudo_rgb=pseudo_rgb)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def dataset_split(self,csv_all_img):
        df= pd.read_csv(csv_all_img,header=0)
        df_train = df[df.split == "train"]
        df_val = df[df.split == "valid"]
        df_test = df[df.split == "test"]
        return df_train,df_val,df_test


