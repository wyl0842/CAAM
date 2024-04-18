import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from tqdm import tqdm

matplotlib.rcParams.update({'font.size': 20})
# plt.rcParams["font.family"] = 'serif'
# plt.rcParams['font.serif'] = ['SimSun']
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['savefig.dpi'] = 300 #图片像素

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_ce_res18_noise0.6_20220513_seed1/cifar10_cifar10_ce_res18_noise0.6_20220513_seed1_symmetric_0.6_1.txt') as fce:
    lines_ce = fce.readlines()

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_mae_res18_noise0.6_20220515_seed1/cifar10_cifar10_mae_res18_noise0.6_20220515_seed1_symmetric_0.6_1.txt') as fmae:
    lines_mae = fmae.readlines()

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_sce1_res18_noise0.6_20220515_seed1/cifar10_cifar10_sce1_res18_noise0.6_20220515_seed1_symmetric_0.6_1.txt') as fsce:
    lines_sce = fsce.readlines()

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacemae15_res18_noise0.6_20220517_seed1/cifar10_cifar10_adacemae15_res18_noise0.6_20220517_seed1_symmetric_0.6_1.txt') as fcemae:
    lines_cemae = fcemae.readlines()

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacence_res18_noise0.6_20220516_seed1/cifar10_cifar10_adacence_res18_noise0.6_20220516_seed1_symmetric_0.6_1.txt') as fcence:
    lines_cence = fcence.readlines()

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenfl10_res18_noise0.6_20220515_seed1/cifar10_cifar10_adacenfl10_res18_noise0.6_20220515_seed1_symmetric_0.6_1.txt') as fcenfl:
    lines_cenfl = fcenfl.readlines()

with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenrce_res18_noise0.6_20220516_seed1/cifar10_cifar10_adacenrce_res18_noise0.6_20220516_seed1_symmetric_0.6_1.txt') as fcenrce:
    lines_cenrce = fcenrce.readlines()

with open('//home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenmae10_res18_noise0.6_20220517_seed1/cifar10_cifar10_adacenmae10_res18_noise0.6_20220517_seed1_symmetric_0.6_1.txt') as fcenmae:
    lines_cenmae = fcenmae.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_ce_res18_noise0.4_20220515_seed1/cifar10_cifar10_ce_res18_noise0.4_20220515_seed1_symmetric_0.4_1.txt') as fce:
#     lines_ce = fce.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_mae_res18_noise0.4_20220519_seed1/cifar10_cifar10_mae_res18_noise0.4_20220519_seed1_symmetric_0.4_1.txt') as fmae:
#     lines_mae = fmae.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_sce_res18_noise0.4_20220519_seed1/cifar10_cifar10_sce_res18_noise0.4_20220519_seed1_symmetric_0.4_1.txt') as fsce:
#     lines_sce = fsce.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacemae_res18_noise0.4_20220519_seed1/cifar10_cifar10_adacemae_res18_noise0.4_20220519_seed1_symmetric_0.4_1.txt') as fcemae:
#     lines_cemae = fcemae.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacence_res18_noise0.4_20220519_seed1/cifar10_cifar10_adacence_res18_noise0.4_20220519_seed1_symmetric_0.4_1.txt') as fcence:
#     lines_cence = fcence.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenfl_res18_noise0.4_20220519_seed1/cifar10_cifar10_adacenfl_res18_noise0.4_20220519_seed1_symmetric_0.4_1.txt') as fcenfl:
#     lines_cenfl = fcenfl.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenrce_res18_noise0.4_20220519_seed1/cifar10_cifar10_adacenrce_res18_noise0.4_20220519_seed1_symmetric_0.4_1.txt') as fcenrce:
#     lines_cenrce = fcenrce.readlines()

# with open('/home/wangyl/Code/CAAM/output/results_cdr/cifar10/cifar10_adacenmae10_res18_noise0.4_20220517_seed1/cifar10_cifar10_adacenmae10_res18_noise0.4_20220517_seed1_symmetric_0.4_1.txt') as fcenmae:
#     lines_cenmae = fcenmae.readlines()

x0 = list(range(1, 41))
x = [i*5 for i in x0]

ce_acc = [0] * 40
mae_acc = [0] * 40
sce_acc = [0] * 40
cemae_acc = [0] * 40
cence_acc = [0] * 40
cenfl_acc = [0] * 40
cenrce_acc = [0] * 40
cenmae_acc = [0] * 40

for i in range(40):
    # if (i+1)%5==0:
    ce_acc[i] = float(lines_ce[2+5*i].split(' ')[3][:])
    mae_acc[i] = float(lines_mae[2+5*i].split(' ')[3][:])
    sce_acc[i] = float(lines_sce[2+5*i].split(' ')[3][:])
    cemae_acc[i] = float(lines_cemae[2+5*i].split(' ')[3][:])
    cence_acc[i] = float(lines_cence[2+5*i].split(' ')[3][:])
    cenfl_acc[i] = float(lines_cenfl[2+5*i].split(' ')[3][:])
    cenrce_acc[i] = float(lines_cenrce[2+5*i].split(' ')[3][:])
    cenmae_acc[i] = float(lines_cenmae[2+5*i].split(' ')[3][:])

plt.figure(figsize=(10,5))
l1, = plt.plot(x, ce_acc, label="CE", linewidth=2)
l2, = plt.plot(x, mae_acc, label="MAE", linewidth=2)
l3, = plt.plot(x, sce_acc, label="SCE", linewidth=2)
l4, = plt.plot(x, cemae_acc, label="CE+MAE", linewidth=2)
l5, = plt.plot(x, cence_acc, label="CE+NCE", linewidth=2)
l6, = plt.plot(x, cenfl_acc, label="CE+NFL", linewidth=2)
l7, = plt.plot(x, cenrce_acc, label="CE+NRCE", linewidth=2)
l8, = plt.plot(x, cenmae_acc, label="CE+NMAE", linewidth=2)
# plt.xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 18})
# plt.ylabel('Test Accuracy', fontdict={'family': 'Times Roman', 'size': 18})
# # plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8], labels=["CE", "MAE", "SCE", "CE+MAE", "CE+NCE", "CE+NFL", "CE+NRCE", "CE+NMAE"], loc=0, prop={'family': 'Times Roman', 'size': 14})
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.tick_params(axis='both', which='minor', labelsize=14)
ax = plt.gca()
# 设置轴的主刻度
# x轴
ax.yaxis.set_major_locator(MultipleLocator(10))  # 设置20倍数
plt.xlabel('Epoch')
plt.ylabel('ACC')
# plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8], labels=["CE", "MAE", "SCE", "CE+MAE", "CE+NCE", "CE+NFL", "CE+NRCE", "CE+NMAE"], bbox_to_anchor=(1.35, 1), labelspacing=0.6, handletextpad=0.2, loc=1)
plt.tick_params(axis='both', which='major')
plt.tick_params(axis='both', which='minor')
plt.grid()
plt.savefig("20240320plot/cifar10_noise0.6_robustloss_0913.jpg", bbox_inches='tight')

# plt.figure(figsize=(10,6))
# plt.plot(x, ce_acc, label="CE")
# plt.plot(x, mae_acc, label="MAE")
# plt.plot(x, sce_acc, label="SCE")
# plt.plot(x, cemae_acc, label="CE+MAE")
# plt.plot(x, cence_acc, label="CE+NCE")
# plt.plot(x, cenfl_acc, label="CE+NFL")
# plt.plot(x, cenrce_acc, label="CE+NRCE")
# plt.plot(x, cenmae_acc, label="CE+NMAE")
# plt.xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 18})
# plt.ylabel('Test Accuracy', fontdict={'family': 'Times Roman', 'size': 18})
# # plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# plt.legend()
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.tick_params(axis='both', which='minor', labelsize=14)
# plt.grid()
# plt.savefig("20240320plot/cifar10_noise0.6_robustloss_0913.jpg", bbox_inches='tight')

# plt.figure(figsize=(10,10))
# fig, ax = plt.subplots(1, 1)
# # ax_sub = ax.twinx()
# # plt.subplot(1,2,1)
# l1, = ax.plot(x, ce_sum, color='r', label="CE fre")
# l2, = ax.plot(x, adrl_sum, color='b', label="ADRL fre")
# l3, = ax_sub.plot(x, ce_acc, color='r', linestyle='--', label="CE acc")
# l4, = ax_sub.plot(x, adrl_acc, color='b', linestyle='--', label="ADRL acc")
# ax.set_xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 18})
# ax.set_ylabel('Number of Activation', fontdict={'family': 'Times Roman', 'size': 18})
# ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# ax_sub.set_ylabel('Test Accuracy', fontdict={'family': 'Times Roman', 'size': 18})
# leg1 = plt.legend(handles=[l1, l2], labels=["CE fre", "ADRL fre"], loc=4, prop={'family': 'Times Roman', 'size': 14})
# plt.gca().add_artist(leg1)
# plt.legend(handles=[l3, l4], labels=["CE acc", "ADRL acc"], loc=1, prop={'family': 'Times Roman', 'size': 14})
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.tick_params(axis='both', which='minor', labelsize=14)
# ax_sub.tick_params(axis='both', which='major', labelsize=14)
# ax_sub.tick_params(axis='both', which='minor', labelsize=14)
# plt.savefig("cifar10_noise0.4_res50_20220515_fre.jpg", bbox_inches='tight')
# # plt.title('tire_res50_20220406_layer4_class0_Frequency',fontsize='large', fontweight='bold')
# # plt.subplot(1,2,2)
# fig, ax = plt.subplots(1, 1)
# ax_sub = ax.twinx()
# l1, = ax.plot(x, mag_ce_sum, color='r', label="CE mag")
# l2, = ax.plot(x, mag_adrl_sum, color='b', label="ADRL mag")
# l3, = ax_sub.plot(x, ce_acc, color='r', linestyle='--', label="CE acc")
# l4, = ax_sub.plot(x, adrl_acc, color='b', linestyle='--', label="ADRL acc")
# ax.set_xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 18})
# ax.set_ylabel('Magnitude of Activation', fontdict={'family': 'Times Roman', 'size': 18})
# ax_sub.set_ylabel('Test Accuracy', fontdict={'family': 'Times Roman', 'size': 18})
# leg2 = plt.legend(handles=[l1, l2], labels=["Clean mag", "Noise mag"], loc=4, prop={'family': 'Times Roman', 'size': 14})
# plt.gca().add_artist(leg2)
# plt.legend(handles=[l3, l4], labels=["Clean acc", "Noise acc"], loc=1, prop={'family': 'Times Roman', 'size': 14})
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.tick_params(axis='both', which='minor', labelsize=14)
# ax_sub.tick_params(axis='both', which='major', labelsize=14)
# ax_sub.tick_params(axis='both', which='minor', labelsize=14)
# # plt.title('tire_res50_20220406_layer4_class0_Magnitude',fontsize='large', fontweight='bold')
# plt.savefig("cifar10_noise0.4_res50_20220515_mag.jpg", bbox_inches='tight')
# # plt.show()

# # 激活频率
# plt.figure(figsize=(10,8))
# l1, = plt.plot(x, ce_sum, color='r', label="CE")
# l2, = plt.plot(x, adrl_sum, color='b', label="ADRL")
# plt.xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 28})
# plt.ylabel('Number of Activation', fontdict={'family': 'Times Roman', 'size': 28})
# plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# plt.legend(handles=[l1, l2], labels=["CE", "ADRL"], loc=1, prop={'family': 'Times Roman', 'size': 24})
# plt.tick_params(axis='both', which='major', labelsize=24)
# plt.tick_params(axis='both', which='minor', labelsize=24)
# plt.grid()
# plt.savefig("cifar10_noise0.4_ceandadrl_fre.jpg", bbox_inches='tight')

# # 激活幅度
# plt.figure(figsize=(10,8))
# l3, = plt.plot(x, mag_ce_sum, color='r', label="CE")
# l4, = plt.plot(x, mag_adrl_sum, color='b', label="ADRL")
# plt.xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 28})
# plt.ylabel('Magnitude of Activation', fontdict={'family': 'Times Roman', 'size': 28})
# # plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# plt.legend(handles=[l3, l4], labels=["CE", "ADRL"], loc=1, prop={'family': 'Times Roman', 'size': 24})
# plt.tick_params(axis='both', which='major', labelsize=24)
# plt.tick_params(axis='both', which='minor', labelsize=24)
# plt.grid()
# plt.savefig("cifar10_noise0.4_ceandadrl_mag.jpg", bbox_inches='tight')

# # 准确率
# plt.figure(figsize=(10,8))
# l5, = plt.plot(x, ce_acc, color='r', label="CE")
# l6, = plt.plot(x, adrl_acc, color='b', label="ADRL")
# plt.xlabel('Epoch', fontdict={'family': 'Times Roman', 'size': 28})
# plt.ylabel('Test Accuracy', fontdict={'family': 'Times Roman', 'size': 28})
# # plt.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
# plt.legend(handles=[l5, l6], labels=["CE", "ADRL"], loc=1, prop={'family': 'Times Roman', 'size': 24})
# plt.tick_params(axis='both', which='major', labelsize=24)
# plt.tick_params(axis='both', which='minor', labelsize=24)
# plt.grid()
# plt.savefig("cifar10_noise0.4_ceandadrl_acc.jpg", bbox_inches='tight')