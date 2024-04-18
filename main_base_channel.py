# python main.py --dataset mnist --noise_type symmetric --noise_rate 0.2 --model_type ?? --seed 1
# CUDA_VISIBLE_DEVICES=1 python ./STD_Train/main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2 --model_type res18_noise0.2_10epoch --seed 1
# CUDA_VISIBLE_DEVICES=1 python ./STD_Train/main.py --dataset cifar10 --noise_type symmetric --noise_rate 0 --model_type res18_noise0 --seed 1
# CUDA_VISIBLE_DEVICES=1 python main_base_channel.py --dataset cifar100 --noise_type symmetric --noise_rate 0.2 --model_type cifar100_caam_res50_noise_20220409_seed1 --seed 1
# CUDA_VISIBLE_DEVICES=1 python main_base_channel.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2 --model_type cifar10_adacemae_res18_noise0.2_20220505_seed1 --seed 1 --model_name robust_loss --loss_name adacerce
# CUDA_VISIBLE_DEVICES=0 python main_base_channel.py --dataset cifar10 --noise_type symmetric --noise_rate 0.6 --model_type cifar10_adacenmae10_9_res18_noise0.6_20220519_seed1_1 --seed 1 --model_name robust_loss --loss_name adacenmae
# CUDA_VISIBLE_DEVICES=1 python main_base_channel.py --dataset cifar10 --noise_type asymmetric --noise_rate 0.4 --model_type cifar10_mae_res18_asnoise0.4_20220524_seed1 --seed 1 --model_name mae
# CUDA_VISIBLE_DEVICES=0 python main_base_channel.py --dataset cifar10 --noise_type asymmetric --noise_rate 0.3 --model_type 20220216_cifar10_ce_res18_asnoise0.3_seed1 --seed 1 --model_name ce

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import LeNet, CifarNet
from util_main import get_activation_values, found_turning_point, update_learning_pace, uniform_noise_model_P, update_trainingloss
from loss import boot_soft, boot_hard, forward, backward, lid_paced_loss
from robust_loss import SCELoss, MeanAbsoluteError, NFLandMAE

from torch.optim.lr_scheduler import MultiStepLR
import dataloader_clothing1M as dataloader
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torch.optim as optim
import argparse, sys
import numpy as np
import datetime
import data_load_base as data_load
import resnet
import tools

import warnings

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=0, help="No.")
parser.add_argument('--d', type=str, default='output', help="description")
parser.add_argument('--p', type=int, default=0, help="print")
parser.add_argument('--c', type=int, default=10, help="class")
parser.add_argument('--lr', type=float, default=0.01)#0.01
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='output/results_cdr/')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate')
parser.add_argument('--dataset', type=str, help='mnist, fmnist, cifar10, cifar100', default='cifar10')
parser.add_argument('--model_name', type=str, help='boot_hard, boot_soft, forward, backward, caam, robust_loss', default='robust_loss')
parser.add_argument('--loss_name', type=str, help='adacemae, adacerce', default='adacerce')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=350)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--model_type', type=str, help='[ce, ours]', default='cdr')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--gpu', type=int, help='ind of gpu', default=0)
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-3)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--batch_size', type=int, help='batch_size', default=32)
parser.add_argument('--train_len', type=int, help='the number of training data', default=54000)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--data_path', default='/data/data_wyl/titan/Dataset/Noisy_label/Clothing1M/images', type=str, help='path to dataset')
args = parser.parse_args()

print(args)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
learning_rate = args.lr

# load dataset
def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset=='fmnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 100
        args.batch_size = 32
        args.num_gradual = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.fmnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.fmnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.fmnist_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)
    
    
    if args.dataset=='mnist':
        args.channel = 1
        args.feature_size = 28 * 28
        args.num_classes = 10
        args.n_epoch = 200
        args.batch_size = 32
        args.num_gradual = 10
        args.train_len = int(60000 * 0.9)
        train_dataset = data_load.mnist_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.mnist_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.mnist_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ),(0.3081, )),]),
                                        target_transform=tools.transform_target)
        
        
    
    if args.dataset=='cifar10':
        args.channel = 3
        args.num_classes = 10
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        args.batch_size = 64
        args.num_gradual = 20
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar10_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.cifar10_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.cifar10_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target)
    
    
    if args.dataset=='cifar100':
        args.channel = 3
        args.num_classes = 100
        args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        args.batch_size = 64
        args.num_gradual = 20
        args.train_len = int(50000 * 0.9)
        train_dataset = data_load.cifar100_dataset(True,
                                        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)

        val_dataset = data_load.cifar100_dataset(False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target,
                                        dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed)


        test_dataset = data_load.cifar100_test_dataset(
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ]),
                                        target_transform=tools.transform_target)
    

    return train_dataset, val_dataset, test_dataset

# load dataset
def load_data_clothing1m(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.batch_size = 8

    loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=4,num_batches=args.num_batches)
    
    if args.dataset=='clothing1m_nl':
        args.channel = 3
        args.num_classes = 14
        # args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        
        args.num_gradual = 20
        args.train_len = args.num_batches*args.batch_size
        # loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=4,num_batches=args.num_batches)
        train_dataset = np.zeros(args.batch_size*args.num_batches)
        val_dataset = np.zeros(14313)
        test_dataset = np.zeros(10526)
        train_loader = loader.run('warmup')
        # train_loader = loader.run('test')
        val_loader = loader.run('val')
        test_loader = loader.run('test')

    if args.dataset=='clothing1m_cl':
        args.channel = 3
        args.num_classes = 14
        # args.feature_size = 3 * 32 * 32
        args.n_epoch = 200
        
        args.num_gradual = 20
        args.train_len = 47570
        # loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=4)
        train_dataset = np.zeros(47570)
        val_dataset = np.zeros(14313)
        test_dataset = np.zeros(10526)
        train_loader = loader.run('clean_train')
        # train_loader = loader.run('test')
        val_loader = loader.run('val')
        test_loader = loader.run('test')
        
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type
save_dir = args.result_dir + args.dataset + '/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)



def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# def train_one_step(net, data, label, optimizer, criterion, nonzero_ratio, clip):
#     net.train()
#     pred = net(data)
#     loss = criterion(pred, label)
#     loss.backward()

#     optimizer.step()
#     optimizer.zero_grad()
#     acc = accuracy(pred, label, topk=(1,))

#     return float(acc[0]), loss


def train(train_loader, epoch, model1, optimizer1, args, criterion):
    model1.train()
    train_total=0
    train_correct=0
    # clip_narry = np.linspace(1-args.noise_rate, 1, num=args.num_gradual)
    # clip_narry = clip_narry[::-1]
    # if epoch < args.num_gradual:
    #     clip = clip_narry[epoch]
   
    # clip = (1 - args.noise_rate)
    for i, (data, labels, _) in enumerate(train_loader):
        # ind=indexes.cpu().numpy().transpose()
        data = data.cuda()
        labels = labels.cuda()
        optimizer1.zero_grad()
        # Forward + Backward + Optimize
        logits1 = model1(data)
        # if found:
        #     # 真实标签的one-hot格式
        #     labels = F.one_hot(labels, num_classes=7)
        #     # 预测标签的one-hot格式
        #     pred_labels = F.one_hot(torch.argmax(logits1, dim=1), num_classes=7)
        #     # 组合成的新label
        #     labels = alpha * labels + (1. - alpha) * pred_labels
        #     labels = torch.argmax(labels, dim=1)
        # criterion = nn.CrossEntropyLoss()
        loss = criterion(logits1, labels)
        loss.backward()
        optimizer1.step()
        # Accuracy
        prec1,  = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        # Loss transfer 

        # prec1, loss = train_one_step(model1, data, labels, optimizer1, nn.CrossEntropyLoss(), clip, clip)
       
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f' 
                  %(epoch+1, args.n_epoch, i+1, args.train_len//args.batch_size, float(prec1[0]), loss.item()))
        
        # del data, labels, logits1, loss
        # torch.cuda.empty_cache()
        
      
    train_acc1=float(train_correct)/float(train_total)
    return train_acc1, loss.item()


# Evaluate the Model
def evaluate(test_loader, model1):
    
    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.cuda()
            logits1 = model1(data)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels.long()).sum()

        acc1 = 100 * float(correct1) / float(total1)

    return acc1


def main(args):
    # Data Loader (Input Pipeline)
    model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.seed)
    txtfile = save_dir + "/" + model_str + ".txt"
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if os.path.exists(txtfile):
        os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))
    
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    if args.dataset in ['mnist', 'fmnist', 'cifar10', 'cifar100']:
        train_dataset, val_dataset, test_dataset = load_data(args)


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                shuffle=True)
        
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                shuffle=False)
        
        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                drop_last=False,
                                                shuffle=False)
    
    else:
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_data_clothing1m(args)
    
    
    # Define models
    print('building model...')
    
    if args.dataset == 'mnist':
        # clf1 = LeNet()
        # clf1 = CifarNet()
        clf1 = resnet.ResNet50(input_channel=1, num_classes=10)
        # clf1 = resnet.ResNet18(input_channel=1, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'fmnist':
        clf1 = resnet.ResNet50(input_channel=1, num_classes=10)
        # clf1 = resnet.ResNet50(input_channel=3, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)
    elif args.dataset == 'cifar10':
        clf1 = resnet.ResNet18(input_channel=3, num_classes=10)
        # clf1 = CifarNet()
        # clf1 = resnet.ResNet50(input_channel=3, num_classes=10)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
    elif args.dataset == 'cifar100':
        clf1 = resnet.ResNet18(input_channel=3, num_classes=100)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
    elif args.dataset == 'clothing1m_nl':
        clf1 = resnet.ResNet50(input_channel=3, num_classes=14)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
    elif args.dataset == 'clothing1m_cl':
        clf1 = resnet.ResNet50(input_channel=3, num_classes=14)
        optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer1, milestones=[40, 80], gamma=0.1)
        
    clf1.cuda()
    
    with open(txtfile, "a") as myfile:
        if args.model_name == 'caam' or args.model_name == 'robust_loss':
        # myfile.write('epoch train_acc1 val_acc1 test_acc1 Loss\n')
            myfile.write('epoch train_acc1 val_acc1 test_acc1 Loss Act_value Turning_epoch\n')
        else:
            myfile.write('epoch train_acc1 val_acc1 test_acc1 Loss\n')

    epoch = 0
    train_acc1 = 0
    if args.model_name == 'boot_hard':
        use_loss = boot_hard
    elif args.model_name == 'boot_soft':
        use_loss = boot_soft
    elif args.model_name == 'forward':
        P = uniform_noise_model_P(args.num_classes, args.noise_rate)
        use_loss = forward(P)
    elif args.model_name == 'backward':
        P = uniform_noise_model_P(args.num_classes, args.noise_rate)
        use_loss = backward(P)
    elif args.model_name == 'mae':
        use_loss = MeanAbsoluteError(num_classes=args.num_classes)
    elif args.model_name == 'sce':
        if args.dataset == 'cifar100':
            use_loss = SCELoss(alpha=6.0, beta=0.1, num_classes=args.num_classes)
        else:
            use_loss = SCELoss(alpha=0.1, beta=1.0, num_classes=args.num_classes)
    elif args.model_name == 'apl':
        if args.dataset == 'cifar100':
            use_loss = NFLandMAE(alpha=10.0, beta=1.0, num_classes=args.num_classes)
        else:
            use_loss = NFLandMAE(alpha=1.0, beta=1.0, num_classes=args.num_classes)
    elif args.model_name == 'caam':
        if args.dataset == 'cifar100':
            use_loss = lid_paced_loss(beta1=6.0, beta2=0.1)
        else:
            use_loss = lid_paced_loss(beta1=0.1, beta2=1.0)
    elif args.model_name == 'robust_loss':
        use_loss = update_trainingloss(0.0, args.num_classes, args.loss_name)
    # elif args.model_name == 'robust_adacencemae':
    #     use_loss = AdaCENCEandMAE(alpha=1.0, beta=1.0, lambda1=1.0, lambda2=0.0, num_classes=args.num_classes)
    # elif args.model_name == 'robust_adacemae':
    #     use_loss = AdaCEandMAE(alpha=1.0, beta=1.0, lambda1=1.0, lambda2=0.0, num_classes=args.num_classes)
    else:
        use_loss = nn.CrossEntropyLoss()
    # use_loss = lid_paced_loss(alpha=1.0)
   

    # evaluate models with random weights
    val_acc1 = evaluate(val_loader, clf1)
    print('Epoch [%d/%d] Val Accuracy on the %s val data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(val_dataset), val_acc1))
    
    test_acc1 = evaluate(test_loader, clf1)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (
    epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
    # save results
    with open(txtfile, "a") as myfile:
        if args.model_name == 'caam' or args.model_name == 'robust_loss':
        # myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + ' ' + str(0) + "\n")
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + ' ' + str(0) + ' ' + str(0) +  ' ' + str(-1) + "\n")
        else:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + ' ' + str(0) + "\n")
    val_acc_list = []
    test_acc_list = []
    
    A_values = []
    turning_epoch = -1
    found = False
    alpha = 1.0
    for epoch in range(0, args.n_epoch):
        scheduler1.step()
        print(optimizer1.state_dict()['param_groups'][0]['lr'])
        if args.model_name == 'caam':
            Activation_value = get_activation_values(clf1, test_loader)
            p_lambda = epoch*1./args.n_epoch

            # deal with possible illegal lid value
            if Activation_value > 0:
                A_values.append(Activation_value)
            else:
                A_values.append(A_values[-1])
            
            # find the turning point where to apply lid-paced learning strategy
            flag, turning_epoch, best_model_path = found_turning_point(A_values, turning_epoch, args)
            if flag:
                print('Turning Point Found!')
                found = True
                
                clf1.load_state_dict(torch.load(best_model_path))
            if found:
                use_loss = update_learning_pace(A_values, turning_epoch, p_lambda)
            #     expansion = A_values[-1] / np.min(A_values)
            #     alpha = np.exp(-p_lambda * expansion)

            if len(A_values) > 5:
                print('A_values = ..., ', A_values[-5:])
            else:
                print('A_values = ..., ', A_values)
        elif args.model_name == 'robust_loss':
            print('Calculate Activation')
            # Activation_value = get_activation_values(net1, val_loader)
            Activation_value = get_activation_values(clf1, test_loader)

            # deal with possible illegal lid value
            if Activation_value > 0:
                A_values.append(Activation_value)
            else:
                A_values.append(A_values[-1])
            
            # find the turning point where to apply lid-paced learning strategy
            flag, turning_epoch, best_model_path = found_turning_point(A_values, turning_epoch, args)
            if flag:
                print('Turning Point Found!')
                found = True
                
                clf1.load_state_dict(torch.load(best_model_path))
            if found:
                expansion = A_values[-1] / np.min(A_values)
                p_lambda = (epoch-turning_epoch)*1./10 * expansion
                use_loss = update_trainingloss(p_lambda, args.num_classes, args.loss_name)
            #     expansion = A_values[-1] / np.min(A_values)
            #     alpha = np.exp(-p_lambda * expansion)

            if len(A_values) > 5:
                print('A_values = ..., ', A_values[-5:])
            else:
                print('A_values = ..., ', A_values)
        clf1.train()
        
        train_acc1, loss_value = train(train_loader, epoch, clf1, optimizer1, args, use_loss)
        val_acc1 = evaluate(val_loader, clf1)
        val_acc_list.append(val_acc1)
        test_acc1 = evaluate(test_loader, clf1)
        test_acc_list.append(test_acc1)
        
        # save results
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %% ' % (
        epoch + 1, args.n_epoch, len(test_dataset), test_acc1))
        with open(txtfile, "a") as myfile:
            # myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + ' ' + str(loss_value) + "\n")
            if args.model_name == 'caam' or args.model_name == 'robust_loss':
                myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + ' ' + str(loss_value) +  ' ' + str(A_values[-1]) +  ' ' + str(turning_epoch) + "\n")
            else:
                myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(val_acc1) + ' ' + str(test_acc1) + ' ' + str(loss_value) + "\n")
        # # 每五代保存一次模型
        # if (epoch+1)%5 == 0:
        #     torch.save(clf1.state_dict(), save_dir + "/" + model_str + '_epoch' + str(epoch) + '.pkl')
        # 每代保存一次模型
        # torch.save(clf1.state_dict(), save_dir + "/" + model_str + '_epoch' + str(epoch) + '.pkl')
    id = np.argmax(np.array(val_acc_list))
    test_acc_max = test_acc_list[id]
    torch.save(clf1.state_dict(), save_dir + "/" + model_str + '.pkl')
    return test_acc_max

if __name__ == '__main__':
    best_acc = main(args)
    
