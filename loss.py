import numpy as np
# from keras import backend as K
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_cross_entropy(alpha, beta):
    """
    Symmetric Cross Entropy: 
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    https://arxiv.org/abs/1908.06112
    """
    def loss(y_pred, y_true):
        y_true = F.one_hot(y_true, num_classes=7)
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        # y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        # y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)
        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)

        return alpha*torch.mean(-torch.sum(y_true_1 * torch.log(y_pred_1), dim = -1)) + beta*torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2), dim = -1))
    return loss

def cross_entropy():
    # return K.categorical_crossentropy(y_true, y_pred)
    return nn.CrossEntropyLoss()


# def boot_soft(y_true, y_pred):
#     """
#     2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
#     https://arxiv.org/abs/1412.6596

#     :param y_true: 
#     :param y_pred: 
#     :return: 
#     """
#     beta = 0.95

#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     return -K.sum((beta * y_true + (1. - beta) * y_pred) *
#                   K.log(y_pred), axis=-1)

# def boot_hard(y_true, y_pred):
#     """
#     2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
#     https://arxiv.org/abs/1412.6596

#     :param y_true: 
#     :param y_pred: 
#     :return: 
#     """
#     beta = 0.8

#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
#     return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
#                   K.log(y_pred), axis=-1)


def boot_soft(y_pred, y_true):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    https://arxiv.org/abs/1412.6596

    :param y_true: 
    :param y_pred: 
    :return: 
    """
    beta = 0.95

    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1])
    y_pred = F.softmax(y_pred, dim=-1)
    y_pred = torch.clamp(y_pred, min=1e-20, max=1.0) # 这步必须要，非常关键，否则训练损失会nan
    y_new = beta * y_true + (1. - beta) * y_pred
    y_pred = torch.log(y_pred)
    return -torch.sum(y_new*y_pred)/y_pred.shape[0]
    # return F.nll_loss(y_pred, y_new)

def boot_hard(y_pred, y_true):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    https://arxiv.org/abs/1412.6596

    :param y_true: 
    :param y_pred: 
    :return: 
    """
    beta = 0.8

    y_true = F.one_hot(y_true, num_classes=y_pred.shape[1])
    y_pred = F.softmax(y_pred, dim=-1)
    y_pred = torch.clamp(y_pred, min=1e-20, max=1.0)
    pred_labels = F.one_hot(torch.argmax(y_pred, dim=-1), num_classes=y_pred.shape[1])
    y_new = beta * y_true + (1. - beta) * pred_labels
    y_pred = torch.log(y_pred)
    # print(-torch.sum(y_new*y_pred)/y_pred.shape[0])
    return -torch.sum(y_new*y_pred)/y_pred.shape[0]
    # return F.nll_loss(y_pred, y_new)

def forward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return: 
    """
    P = torch.Tensor(P).cuda()
    # P = P.cuda()

    def loss(y_pred, y_true):
        y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
        y_pred = F.softmax(y_pred, dim=-1)
        y_pred = torch.clamp(y_pred, min=1e-20, max=1.0)
        # return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)
        # print(y_pred)
        # print(torch.mm(y_pred, P))
        # print(torch.log(torch.mm(y_pred, P)))
        return -torch.sum(y_true*torch.log(torch.mm(y_pred, P)))/y_pred.shape[0]
        # return -torch.sum(y_true*torch.log(y_pred))/y_pred.shape[0]

    return loss


def backward(P):
    """
    Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    CVPR17 https://arxiv.org/abs/1609.03683
    :param P: noise model, a noisy label transition probability matrix
    :return: 
    """
    P_inv = torch.Tensor(np.linalg.inv(P)).cuda()
    # P = torch.Tensor(P).cuda()
    # P = P.cuda()

    def loss(y_pred, y_true):
        y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
        y_pred = F.softmax(y_pred, dim=-1)
        y_pred = torch.clamp(y_pred, min=1e-20, max=1.0)
        y_new = torch.mm(y_true, P_inv)
        return -torch.sum(y_new*torch.log(y_pred))/y_pred.shape[0]

    return loss

# def forward(P):
#     """
#     Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
#     CVPR17 https://arxiv.org/abs/1609.03683
#     :param P: noise model, a noisy label transition probability matrix
#     :return: 
#     """
#     P = K.constant(P)

#     def loss(y_true, y_pred):
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#         return -K.sum(y_true * K.log(K.dot(y_pred, P)), axis=-1)

#     return loss


# def backward(P):
#     """
#     Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
#     CVPR17 https://arxiv.org/abs/1609.03683
#     :param P: noise model, a noisy label transition probability matrix
#     :return: 
#     """
#     P_inv = K.constant(np.linalg.inv(P))

#     def loss(y_true, y_pred):
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#         return -K.sum(K.dot(y_true, P_inv) * K.log(y_pred), axis=-1)

#     return loss


# def lid(logits, k=20):
#     """
#     Calculate LID for each data point in the array.

#     :param logits:
#     :param k: 
#     :return: 
#     """
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
#     v_log = tf.reduce_sum(tf.log(m + K.epsilon()), axis=1)  # to avoid nan
#     lids = -k / v_log

#     return lids


def lid_paced_loss(alpha=1.0, beta1=0.1, beta2=1.0):
    """TO_DO
    Class wise lid pace learning, targeting classwise asymetric label noise.

    Args:      
      alpha: lid based adjustment paramter: this needs real-time update.
    Returns:
      Loss tensor of type float.
    """
    eps = 1e-7
    if alpha == 1.0:
        # return symmetric_cross_entropy(alpha=beta1, beta=beta2)
        return nn.CrossEntropyLoss()
    else:
        def loss(y_pred, y_true):
            # 真实标签的one-hot格式
            y_true = F.one_hot(y_true, num_classes=y_pred.shape[1])
            # 预测标签的one-hot格式
            y_pred = F.softmax(y_pred, dim=-1)
            pred_labels = F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1])
            # 组合成的新label
            y_new = alpha * y_true + (1. - alpha) * pred_labels
            # 预测值归一化
            # print(y_pred.shape)
            # print(torch.sum(y_pred, dim=-1).shape)
            y_pred = torch.log(y_pred)
            # y_pred /= torch.sum(y_pred, dim=-1, keepdim=True)
            # y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
            # return torch.mean(-torch.sum(y_new * torch.log(y_pred), dim=-1))
            return -torch.sum(y_new*y_pred)/y_pred.shape[0]
            # return F.nll_loss(y_pred, y_new)

        return loss
