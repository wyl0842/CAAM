import numpy as np
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from PIL import Image
import tools
import torch
from random import choice
import random 

class CustomImageFolder(ImageFolder):
    """
    """
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self)) #该文件夹中的长度
        self.transform0=self.transform
        
    def __getitem__(self, index1):    
        path1 = self.imgs[index1][0] #此时的imgs等于samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label1 = self.imgs[index1][1]

        img0 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform0(img0)
        return img1, label1, path1

class train_dataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, noise_type='symmetric',
                                 noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100):            
        super(train_dataset, self).__init__(root, transform)
        
        # self.indices = range(len(self)) #该文件夹中的长度
        self.transform = transform
        self.target_transform = target_transform
        self.noise_rate = noise_rate
        self.num_class = num_class
        self.random_seed = random_seed
        self.split_per = split_per
        self.train = train
        self.path = [item[0] for item in self.imgs]
        self.label = [item[1] for item in self.imgs]
        self.img = [self.loader(path_item) for path_item in self.path]

        # 为标签加噪声
        # self.noisy_labels = np.array(self.label)
        self.noisy_labels = self.label
        probs_to_change = torch.randint(100, (len(self.noisy_labels),))
        idx_to_change = probs_to_change >= (100.0 - 100 * self.noise_rate)
        for n, _ in enumerate(self.noisy_labels):
            if idx_to_change[n] == 1:
                set_labels = list(set(range(self.num_class)))  # this is a set with the available labels (with the current label)
                set_index = np.random.randint(len(set_labels))
                self.noisy_labels[n] = set_labels[set_index]
        
        # 划分数据集9：1
        num_samples = len(self.noisy_labels)
        np.random.seed(self.random_seed)
        train_set_index = np.random.choice(num_samples, int(num_samples*self.split_per), replace=False)
        all_index = np.arange(num_samples)
        val_set_index = np.delete(all_index, train_set_index)

        self.train_data, self.train_labels, self.val_data, self.val_labels = [], [], [], []
        for idx in train_set_index:
            self.train_data.append(self.img[idx])
            self.train_labels.append(self.noisy_labels[idx])

        for idx in val_set_index:
            self.val_data.append(self.img[idx])
            self.val_labels.append(self.noisy_labels[idx])

    def __getitem__(self, index):
        
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)
        # return len(self.imgs)

class test_dataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, dataset='cifar100', noise_type='symmetric',
                                 noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100):
            
        super(test_dataset, self).__init__(root, transform)
        self.indices = range(len(self)) #该文件夹中的长度
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
           
        path = self.imgs[index][0] #此时的imgs等于samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label = self.imgs[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
     
        return img, label, index
    def __len__(self):
        return len(self.imgs)