import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
from random import choice
import random 
from numpy.testing import assert_array_almost_equal

def multiclass_noisify(y, P, random_state=1):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
#    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    actual_noise = 0

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
#        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
#    print (P)

    return y_train, actual_noise, P

def dataset_split(train_images, train_labels, dataset='mnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    
    clean_train_labels = train_labels[:, np.newaxis] # (50000, 1)
    
    if noise_type == 'symmetric':
         noisy_labels, real_noise_rate, transition_matrix = noisify_multiclass_symmetric(clean_train_labels,  # (50000,1)
                                                                                               noise=noise_rate, 
                                                                                               random_state=random_seed, 
                                                                                               nb_classes=num_classes)  

    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_labels_clean, val_labels_clean = train_labels[train_set_index], train_labels[val_set_index]
    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index] # (45000,)

    return train_set, val_set, train_labels, val_labels, train_labels_clean, val_labels_clean
    
class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, dataset='cifar10', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/cifar10/train_images.npy')
        original_labels = np.load('data/cifar10/train_labels.npy')
        

        # clean images and noisy labels (training and validation)
        self.train_data, _, self.train_labels, _, self.clean_labels_train, _ = dataset_split(original_images, 
                                                                             original_labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)

        if self.train:      
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        
        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))
        
    def __getitem__(self, index):
           
        if self.train:
            img, label, clean_label = self.train_data[index], self.train_labels[index], self.clean_labels_train[index]
            
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            clean_label = self.target_transform(clean_label)
     
        return img, label, clean_label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)

    