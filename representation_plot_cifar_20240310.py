"""
Date: 28/07/2017
feature exploration and visualization

Author: Xingjun Ma
"""
import os
import numpy as np
import torch
import resnet
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import data_load_base_tsne as data_load
import torchvision.transforms as transforms
from loss import cross_entropy

matplotlib.rcParams.update({'font.size': 24})
# plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 300 #图片像素
np.random.seed(1234)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target

train_dataset = data_load.cifar10_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=transform_target,
                                        dataset='cifar10',
                                        noise_type='symmetric',
                                        noise_rate=0.6,
                                        split_per=0.9,
                                        random_seed=1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=128,
                                            num_workers=4,
                                            shuffle=False)

clf1 = resnet.ResNet18(input_channel=3, num_classes=10)
clf2 = resnet.ResNet18(input_channel=3, num_classes=10)
clf1.load_state_dict(torch.load('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_ce_res18_noise0.6_20220513_seed1/cifar10_cifar10_ce_res18_noise0.6_20220513_seed1_symmetric_0.6_1.pkl', map_location=lambda storage, loc: storage))
clf2.load_state_dict(torch.load('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenmae10_res18_noise0.6_20220517_seed1/cifar10_cifar10_adacenmae10_res18_noise0.6_20220517_seed1_symmetric_0.6_1.pkl', map_location=lambda storage, loc: storage))
clf1.cuda()
clf2.cuda()

feat_result_output_ce = []
def get_features_hook(module, data_input, data_output):
        # feat_result_input.append(data_input)
        feat_result_output_ce.append(data_output)

def feature_visualization():
    """
    This is to show how features of incorretly labeled images are overffited to the wrong class.
    plot t-SNE 2D-projected deep features (right before logits).
    This will generate 3 plots in a grid (3x1). 
    The first shows the raw features projections of two classes of images (clean label + noisy label)
    The second shows the deep features learned by cross-entropy after training.
    The third shows the deep features learned using a new loss after training.
    
    :param model_name: a new model other than crossentropy(ce), can be: boot_hard, boot_soft, forward, backward, lid
    :param dataset: 
    :param num_classes:
    :param noise_type;
    :param noise_ratio: 
    :param epochs: to find the last epoch
    :param n_samples: 
    :return: 
    """
    # print('Dataset: %s, model_name: ce/%s, noise ratio: %s%%' % (model_name, dataset, noise_ratio))
    # features_ce = np.array([None, None])
    # features_other = np.array([None, None])
    clean_targets_list = []
    noisy_targets_list = []
    outputs_list = []

    # # load pre-saved to avoid recomputing
    # feature_tmp = "lof/representation_%s_%s.npy" % (dataset, noise_ratio)
    # if os.path.isfile(feature_tmp):
    #     data = np.load(feature_tmp)
    #     features_input = data[0]
    #     features_ce = data[1]
    #     features_other = data[2]
    #
    #     plot(model_name, dataset, noise_ratio, features_input, features_ce, features_other)
    #     return

    # load data
    # X_train, Y_train, X_test, Y_test = get_data(dataset)
    # Y_noisy = np.load("data/noisy_label_%s_%s.npy" % (dataset, noise_ratio))
    # Y_noisy = Y_noisy.reshape(-1)
    
    with torch.no_grad():
        handle = clf2.avgpool.register_forward_hook(get_features_hook)
        for idx, (inputs, targets, clean_targets, _) in enumerate(train_loader):
            feat_result_output_ce.clear()
            inputs = inputs.cuda()
            targets = targets.cuda()
            clean_targets = clean_targets.cuda()
            targets_np = targets.data.cpu().numpy()
            clean_targets_np = clean_targets.data.cpu().numpy()

            outputs = clf2(inputs)
            feat_ce = feat_result_output_ce[0]
            feat_out_ce = feat_ce.view(feat_ce.size(0), -1)
            feat_out_ce_np = feat_out_ce.data.cpu().numpy()
            # outputs_np = outputs.data.cpu().numpy()
            
            noisy_targets_list.append(targets_np[:, np.newaxis])
            clean_targets_list.append(clean_targets_np[:, np.newaxis])
            outputs_list.append(feat_out_ce_np)
            
            if ((idx+1) % 10 == 0) or (idx+1 == len(train_loader)):
                print(idx+1, '/', len(train_loader))
            if idx > 20:
                break

    cleantargets = np.concatenate(clean_targets_list, axis=0)
    noisytargets = np.concatenate(noisy_targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return cleantargets, noisytargets, outputs

def tsne_plot(cleantargets, noisytargets, outputs):
    print('generating t-SNE plot...')
    # tsne_output = bh_sne(outputs)
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['True'] = cleantargets
    df['Noisy'] = noisytargets

    # sns.set(font_scale = 2)

    plt.rcParams['figure.figsize'] = 10, 10
    ax = sns.scatterplot(
        x='x', y='y',
        hue='True',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )
 
    # plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
    # plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
    # ax.legend(bbox_to_anchor=(0.5, -0.2), columnspacing=0.2, loc=8, ncol=10, borderpad=0.2, labelspacing=0.2, handlelength=1, handletextpad=0, title='True Label')
    ax.legend(bbox_to_anchor=(0.5, -0.15), columnspacing=0.8, loc=8, ncol=10, borderpad=0.2, labelspacing=0.4, handlelength=1, handletextpad=0, title='True Label')
    # ax.legend(title='真实标签')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig('20240407plot/tsne_clean_adrl_0913.png', bbox_inches='tight')
    # plt.savefig('20240407plot/tsne_clean_ce_0913.png', bbox_inches='tight')


    # plt.rcParams['figure.figsize'] = 10, 10
    # ax = sns.scatterplot(
    #     x='x', y='y',
    #     hue='Noisy',
    #     palette=sns.color_palette("hls", 10),
    #     data=df,
    #     marker='o',
    #     legend="full",
    #     alpha=0.5
    # )

    # # plt.setp(ax.get_legend().get_texts(), fontsize='18') # for legend text
    # # plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
    # ax.legend(bbox_to_anchor=(0.5, -0.2), columnspacing=0.2, loc=8, ncol=10, borderpad=0.2, labelspacing=0.2, handlelength=1, handletextpad=0, title='噪声标签')

    # plt.xticks([])
    # plt.yticks([])
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.savefig('20240407plot/tsne_noisy_adrl_0913.png', bbox_inches='tight')
    # # plt.savefig('20240310plot/tsne_noisy_ce_0913.png', bbox_inches='tight')

    print('done!')

if __name__ == "__main__":
    cleantargets, noisytargets, outputs = feature_visualization()
    tsne_plot(cleantargets, noisytargets, outputs)