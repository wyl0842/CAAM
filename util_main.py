from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import torch
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import MinMaxScaler
# import keras.backend as K
from scipy.spatial.distance import pdist, cdist, squareform
# from keras.callbacks import ModelCheckpoint, Callback
# from keras.callbacks import LearningRateScheduler
# import tensorflow as tf
from loss import lid_paced_loss
from robust_loss import AdaCENCEandMAE, AdaCEandMAE, AdaCEandNMAE, AdaCEandRCE, AdaCEandNRCE, AdaCEandNCE, AdaCEandNFL
from tqdm import tqdm
import torch.nn as nn

# Set random seed
np.random.seed(123)


# def lid(logits, k=20):
#     """
#     Calculate LID for a minibatch of training samples based on the outputs of the network.

#     :param logits:
#     :param k: 
#     :return: 
#     """
#     epsilon = 1e-12
#     batch_size = tf.shape(logits)[0]
#     # n_samples = logits.get_shape().as_list()
#     # calculate pairwise distance
#     r = tf.reduce_sum(logits * logits, 1)
#     # turn r into column vector
#     r1 = tf.reshape(r, [-1, 1])
#     D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
#         tf.ones([batch_size, batch_size])

#     # find the k nearest neighbor
#     D1 = -tf.sqrt(D)
#     D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
#     D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

#     m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
#     v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
#     lids = -k / v_log
#     return lids


def mle_single(data, x, k):
    """
    lid of a single query point x.
    numpy implementation.

    :param data: 
    :param x: 
    :param k: 
    :return: 
    """
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]


def mle_batch(data, batch, k):
    """
    lid of a batch of query points X.
    numpy implementation.

    :param data: 
    :param batch: 
    :param k: 
    :return: 
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


# def get_lids_random_batch(model, X, k=20, batch_size=128):
#     """
#     Get the local intrinsic dimensionality of each Xi in X_adv
#     estimated by k close neighbours in the random batch it lies in.
#     :param model: if None: lid of raw inputs, otherwise LID of deep representations 
#     :param X: normal images 
#     :param k: the number of nearest neighbours for LID estimation  
#     :param batch_size: default 100
#     :return: lids: LID of normal images of shape (num_examples, lid_dim)
#             lids_adv: LID of advs images of shape (num_examples, lid_dim)
#     """
#     if model is None:
#         lids = []
#         n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#         for i_batch in range(n_batches):
#             start = i_batch * batch_size
#             end = np.minimum(len(X), (i_batch + 1) * batch_size)
#             X_batch = X[start:end].reshape((end - start, -1))

#             # Maximum likelihood estimation of local intrinsic dimensionality (LID)
#             lid_batch = mle_batch(X_batch, X_batch, k=k)
#             lids.extend(lid_batch)

#         lids = np.asarray(lids, dtype=np.float32)
#         return lids

#     # get deep representations
#     funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
#              for out in [model.get_layer("lid").output]]
#     lid_dim = len(funcs)

#     #     print("Number of layers to estimate: ", lid_dim)

#     def estimate(i_batch):
#         start = i_batch * batch_size
#         end = np.minimum(len(X), (i_batch + 1) * batch_size)
#         n_feed = end - start
#         lid_batch = np.zeros(shape=(n_feed, lid_dim))
#         for i, func in enumerate(funcs):
#             X_act = func([X[start:end], 0])[0]
#             X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

#             # Maximum likelihood estimation of local intrinsic dimensionality (LID)
#             lid_batch[:, i] = mle_batch(X_act, X_act, k=k)

#         return lid_batch

#     lids = []
#     n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#     for i_batch in range(n_batches):
#         lid_batch = estimate(i_batch)
#         lids.extend(lid_batch)

#     lids = np.asarray(lids, dtype=np.float32)

#     return lids

######## 获取当前epoch的激活
def get_activation_values(model, t_loader):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model: if None: lid of raw inputs, otherwise LID of deep representations 
    :param X: normal images 
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    feat_result_output = []
    ######## hook函数
    def get_features_hook(module, data_input, data_output):
        # feat_result_input.append(data_input)
        feat_result_output.append(data_output)
    with torch.no_grad():
        # features = list(model.children())[:-2]#去掉池化层及全连接层
        # #print(list(model.children())[:-2])
        # feature_map = nn.Sequential(*features)
        statis_results_robust = 0.
        magnitude_robust = 0.
        count_samples = 0
        batch_idx = 0
        handle = model.layer4.register_forward_hook(get_features_hook)       # 取对抗样本激活值
        for data, label, _ in tqdm(t_loader):
            # 对抗样本计算激活幅值和频率
            feat_result_output.clear()
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            idx = np.where(label.cpu().numpy() == np.array([0]*data.shape[0]))[0]
            idx = torch.tensor(idx)
            count_samples += len(idx)
            if len(idx) > 0:
                feat2 = feat_result_output[0]
                feat_out = feat2[idx]
                # feat_out = feature_map(data)
                if len(feat_out.shape) == 4:
                    N, C, H, W = feat_out.shape
                    feat_out = feat_out.view(N, C, H * W)
                    feat_out = torch.mean(feat_out, dim=-1)
                N, C = feat_out.shape
                max_value = torch.max(feat_out, dim=1, keepdim=True)[0]
                threshold = 1e-2 * max_value
                mask = feat_out > threshold.expand(N, C)
                count_activate = torch.sum(mask, dim=0).view(C)
                feat_mean_magnitude = torch.sum(feat_out, dim=0).view(C)
                for k in range(C):
                    # if feat_mean_magnitude[k] != 0:
                    if count_activate[k] != 0:
                        feat_mean_magnitude[k] = feat_mean_magnitude[k] / count_activate[k].float()
                count_activate = count_activate.cpu().numpy()
                feat_mean_magnitude = feat_mean_magnitude.cpu().numpy()
                if batch_idx == 0:
                    statis_results_robust = count_activate
                    magnitude_robust = feat_mean_magnitude
                else:
                    statis_results_robust = (statis_results_robust + count_activate)
                    magnitude_robust = (magnitude_robust + feat_mean_magnitude) / 2
            batch_idx += 1
        # 频率
        statis_results_robust = np.array(statis_results_robust)
        # 幅值
        magnitude_results_robust = np.array(magnitude_robust)
        # A_value = magnitude_results_robust.sum()
        A_value = statis_results_robust.sum()
        handle.remove()

    return A_value

######## 根据当前代的激活寻找转折点
def found_turning_point(lids, turning_epoch, args):
    init_epoch = 20
    epoch_win = 5
    if len(lids) > init_epoch + epoch_win: #
        if turning_epoch > -1: # if turning point is already found, stop checking
            return False, turning_epoch, ''
        else:
            smooth_lids = lids[-epoch_win-1:-1]
            # if lids[-1] - np.mean(smooth_lids) > 2*np.std(smooth_lids):
            if lids[-epoch_win-1] == np.min(smooth_lids):
                turning_epoch = len(lids) - 2
                min_model_path = args.result_dir + args.dataset + '/%s/' % args.model_type + args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.seed) + '_epoch' + str(turning_epoch) + '.pkl'
                return True, turning_epoch, min_model_path
            else:
                return False, -1, ''
    else:
        return False, -1, ''

######## 调整学习策略
def update_learning_pace(A_values, turning_epoch, p_lambda):
    # # this loss is not working for d2l learning, somehow, why???
    # expansion = A_values[-1] / np.min(A_values)
    expansion = A_values[-1] / np.min(A_values[1:])
    # expansion = A_values[-1] / A_values[turning_epoch]
    alpha = np.exp(-p_lambda * expansion)
    # self.alpha = np.exp(-0.1*expansion)

    # print('## Turning epoch: %s, lambda: %.2f, expansion: %.2f, alpha: %.2f' %
    #         (turning_epoch, self.p_lambda, expansion, self.alpha))

    # self.alpha = np.exp(-expansion)
    return lid_paced_loss(alpha)
    # self.model.compile(loss=lid_paced_loss(self.alpha),
    #                     optimizer=self.model.optimizer, metrics=['accuracy'])

# def update_trainingloss(p_lambda, num_class):
#     if p_lambda > 1:
#         p_lambda = 1.0
#     return AdaCENCEandMAE(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)

def update_trainingloss(p_lambda, num_class, loss_name):
    if p_lambda > 1:
        p_lambda = 1.0
    if loss_name == 'adacemae':
        return AdaCEandMAE(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)
    elif loss_name == 'adacerce':
        return AdaCEandRCE(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)
    elif loss_name == 'adacenmae':
        return AdaCEandNMAE(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)
    elif loss_name == 'adacenrce':
        return AdaCEandNRCE(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)
    elif loss_name == 'adacence':
        return AdaCEandNCE(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)
    elif loss_name == 'adacenfl':
        return AdaCEandNFL(alpha=1.0, beta=1.0, lambda1=1.0-p_lambda, lambda2=p_lambda, num_classes=num_class)

# def get_lr_scheduler(dataset):
#     """
#     customerized learning rate decay for training with clean labels.
#      For efficientcy purpose we use large lr for noisy data.
#     :param dataset: 
#     :param noise_ratio:
#     :return: 
#     """
#     if dataset in ['mnist', 'svhn']:
#         def scheduler(epoch):
#             if epoch > 40:
#                 return 0.001
#             elif epoch > 20:
#                 return 0.01
#             else:
#                 return 0.1

#         return LearningRateScheduler(scheduler)
#     elif dataset in ['cifar-10']:
#         def scheduler(epoch):
#             if epoch > 80:
#                 return 0.001
#             elif epoch > 40:
#                 return 0.01
#             else:
#                 return 0.1

#         return LearningRateScheduler(scheduler)
#     elif dataset in ['cifar-100']:
#         def scheduler(epoch):
#             if epoch > 120:
#                 return 0.001
#             elif epoch > 80:
#                 return 0.01
#             else:
#                 return 0.1

#         return LearningRateScheduler(scheduler)


def uniform_noise_model_P(num_classes, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (num_classes - 1).
    """

    assert (noise >= 0.) and (noise <= 1.)

    P = noise / (num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(P, (1 - noise) * np.ones(num_classes))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


# def get_deep_representations(model, X, batch_size=128):
#     """
#     Get the deep representations before logits.
#     :param model:
#     :param X:
#     :param batch_size:
#     :return:
#     """
#     # last hidden layer is always at index -4
#     output_dim = model.layers[-3].output.shape[-1].value
#     get_encoding = K.function(
#         [model.layers[0].input, K.learning_phase()],
#         [model.layers[-3].output]
#     )

#     n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#     output = np.zeros(shape=(len(X), output_dim))
#     for i in range(n_batches):
#         output[i * batch_size:(i + 1) * batch_size] = \
#             get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

#     return output
