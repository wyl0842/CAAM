from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random
import numpy as np
from PIL import Image
import json
import torch

class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}            
        
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'all':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    # img_path = '%s/'%self.root+l
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)       
        elif self.mode == "labeled":   
            train_imgs = paths 
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
        elif self.mode == "unlabeled":  
            train_imgs = paths 
            pred_idx = (1-pred).nonzero()[0]  
            self.train_imgs = [train_imgs[i] for i in pred_idx]                
            self.probability = [probability[i] for i in pred_idx]            
            print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))                                    
                         
        elif mode=='clean_train':
            self.cltrain_imgs = []
            with open('%s/clean_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    # img_path = '%s/'%self.root+l
                    self.cltrain_imgs.append(img_path)
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    # img_path = '%s/'%self.root+l
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    # img_path = '%s/'%self.root+l
                    self.val_imgs.append(img_path)
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, img_path        
        elif self.mode=='clean_train':
            img_path = self.cltrain_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target, img_path
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        if self.mode=='clean_train':
            return len(self.cltrain_imgs)
        else:
            return len(self.train_imgs)            
        
class clothing_dataloader():  
    def __init__(self, root, batch_size, num_batches, num_workers):    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
                   
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])        
    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='all',num_samples=self.num_batches*self.batch_size)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        elif mode=='train':
            labeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='all',num_samples=self.num_batches*self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
        elif mode=='clean_train':
            clean_dataset = clothing_dataset(self.root,transform=self.transform_train, mode='clean_train')
            clean_loader = DataLoader(
                dataset=clean_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)          
            return clean_loader
        elif mode=='test':
            test_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                # batch_size=1000,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader             
        elif mode=='val':
            val_dataset = clothing_dataset(self.root,transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset, 
                # batch_size=1000,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)             
            return val_loader     


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
