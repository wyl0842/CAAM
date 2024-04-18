# CUDA_VISIBLE_DEVICES=0 python Train_myDataset_channel.py --id tire_res50_caam_20220405_noise0.2
from __future__ import print_function
from cProfile import label
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import random
import os
import argparse
import numpy as np
import dataloader_myDataset as data_load
from util import uniform_noise_model_P, get_activation_values, found_turning_point, update_learning_pace
# from loss import cross_entropy, boot_soft, boot_hard, forward, backward, lid_paced_loss
from loss import cross_entropy, lid_paced_loss
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser(description='PyTorch myDataset Training')
parser.add_argument('--batch_size', default=4, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--noise_rate', type=float, help='overall corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--split_percentage', type=float, help='train and validation', default=0.9)
parser.add_argument('--dataset', type=str, help='mnist, fmnist, cifar10, cifar100', default='tire')
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='clothing1m')
# parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--data_path', default='data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)#123
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_classes', default=7, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def warmup(net, optimizer, dataloader, found, alpha=1.0):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)
        if found:
            # 真实标签的one-hot格式
            labels = F.one_hot(labels, num_classes=7)
            # 预测标签的one-hot格式
            pred_labels = F.one_hot(torch.argmax(outputs, dim=1), num_classes=7)
            # 组合成的新label
            labels = alpha * labels + (1. - alpha) * pred_labels
            labels = torch.argmax(labels, dim=1)
        
        loss = CEloss(outputs, labels)
      
        loss.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        # sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f'
        #         %(batch_idx+1, args.num_batches, loss.item()))
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f'
                %(batch_idx+1, len(train_dataset)/args.batch_size, loss.item()))
        sys.stdout.flush()

# load dataset
def load_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset=='tire':
        args.channel = 1
        args.num_classes = 7
        args.feature_size = 1 * 256 * 256
        args.n_epoch = 200
        # args.batch_size = 128
        args.train_len = int(1700 * 7 * 0.7)
        train_dataset = data_load.train_dataset(os.path.join(args.data_path, 'train'),
                                        True,
                                        transform = transforms.Compose([
                                        # transforms.Resize(input_size),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,], [0.5,]),
                                        transforms.RandomCrop(256, padding=4),
                                        transforms.RandomHorizontalFlip(),                                       
                                        ]),
                                        # target_transform=tools.transform_target,
                                        # dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed,
                                        num_class=args.num_classes)

        val_dataset = data_load.train_dataset(os.path.join(args.data_path, 'train'),
                                        False,
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Grayscale(),
                                        transforms.Normalize([0.5,], [0.5,]),
                                        ]),
                                        # target_transform=tools.transform_target,
                                        # dataset=args.dataset,
                                        noise_type=args.noise_type,
                                        noise_rate=args.noise_rate,
                                        split_per=args.split_percentage,
                                        random_seed=args.seed,
                                        num_class=args.num_classes)


        test_dataset = data_load.test_dataset(os.path.join(args.data_path, 'test'),
                                        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Grayscale(),
                                        transforms.Normalize([0.5,], [0.5,]),
                                        ]),
                                        # target_transform=tools.transform_target
                                        )

        te_dataset = data_load.train_dataset(os.path.join(args.data_path, 'train'),
                                        True,
                                        transform = transforms.Compose([
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5,], [0.5,]),                                      
                                        ]),
                                        noise_rate=0,
                                        split_per=0.9,
                                        random_seed=args.seed,
                                        num_class=7)
    

    return train_dataset, val_dataset, test_dataset, te_dataset

def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Epoch %d  Acc: %.2f%%" %(k,acc))  
    save_point = './checkpoint/%s/%s_net%d.pth.tar'%(args.id,args.id,k)
    torch.save(net.state_dict(), save_point)
    # if acc > best_acc[k-1]:
    #     best_acc[k-1] = acc
    #     print('| Saving Best Net%d ...'%k)
    #     save_point = './checkpoint/%s_net%d.pth.tar'%(args.id,k)
    #     torch.save(net.state_dict(), save_point)
    return acc

def test(net1,test_loader):
    net1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)                 
            _, predicted = torch.max(outputs1, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc
               
def create_model():
    model = models.resnet50(pretrained=False)
    # print(model.conv1)
    model.conv1= nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(2048, args.num_classes)
    model = model.cuda()
    return model     

log_dir = './checkpoint/%s'%args.id
if not os.path.exists(log_dir):
    os.system('mkdir -p %s' % log_dir)
log=open('./checkpoint/%s/%s.txt'%(args.id,args.id),'w')     
log.flush()

print('| Loading dataset')
train_dataset, val_dataset, test_dataset, te_dataset = load_data(args)
# loader = data_load.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        drop_last=True,
                                        shuffle=True)
    
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        drop_last=True,
                                        shuffle=False)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        drop_last=True,
                                        shuffle=False)

te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=128, shuffle=False, num_workers=4)

print('| Building net')
net1 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

# 返回每个样本的loss                      
# CE = nn.CrossEntropyLoss(reduction='none')
# 返回样本loss的平均值
CEloss = nn.CrossEntropyLoss()
# if model_name == 'forward':
#     P = uniform_noise_model_P(num_classes, noise_ratio / 100.)
#     use_loss = forward(P)
# elif model_name == 'backward':
#     P = uniform_noise_model_P(num_classes, noise_ratio / 100.)
#     use_loss = backward(P)
# elif model_name == 'boot_hard':
#     use_loss = boot_hard
# elif model_name == 'boot_soft':
#     use_loss = boot_soft
# elif model_name == 'd2l':
#     if dataset == 'cifar-100':
#         use_loss = lid_paced_loss(beta1=6.0, beta2=0.1)
#     else:
#         use_loss = lid_paced_loss(beta1=0.1, beta2=1.0)
# else:
#     use_loss = cross_entropy

# use_loss = lid_paced_loss(alpha=1.0, beta1=0.1, beta2=1.0)
# use_loss = nn.CrossEntropyLoss()

A_values = []
turning_epoch = -1
found = False
alpha = 1.0
best_acc = [0,0]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr        

    #########
    print('Calculate Activation')
    # Activation_value = get_activation_values(net1, te_loader)
    Activation_value = get_activation_values(net1, te_loader)
    p_lambda = epoch*1./args.num_epochs

    # deal with possible illegal lid value
    if Activation_value > 0:
        A_values.append(Activation_value)
    else:
        A_values.append(A_values[-1])
    
    # find the turning point where to apply lid-paced learning strategy
    flag, turning_epoch, best_model_path = found_turning_point(A_values, turning_epoch, args.id)
    if flag:
        print('Turning Point Found!')
        found = True
        # use_loss = update_learning_pace(A_values, p_lambda)
        net1.load_state_dict(torch.load(best_model_path))
    if found:
        expansion = A_values[-1] / np.min(A_values)
        alpha = np.exp(-p_lambda * expansion)

    if len(A_values) > 5:
        print('A_values = ..., ', A_values[-5:])
    else:
        print('A_values = ..., ', A_values)
    #########

    # train_loader = loader.run('warmup')
    print('Warmup Net1')

    warmup(net1, optimizer1, train_loader, found, alpha)                      
    
    # val_loader = loader.run('val') # validation
    acc1 = val(net1, val_loader, epoch) 
    acc = test(net1, test_loader) 
    # log.write('Validation Epoch:%d      Acc1:%.2f    Activation Value:%d\n'%(epoch, acc1, A_values[-1]))
    log.write('Validation Epoch:%d      Acc1:%.2f      Test Accuracy:%.2f      Act_value:%.2f      Turning Epoch:%d\n'%(epoch, acc1, acc, A_values[-1], turning_epoch))
    # log.write('Test Accuracy:%.2f\n'%(acc))
    log.flush()

# log.write('Turning Epoch:%d\n'%(turning_epoch))
# test_loader = loader.run('test')
# net1.load_state_dict(torch.load('./checkpoint/%s/%s_net%s.pth.tar'%(args.id,args.id,args.num_epochs)))
# acc = test(net1, test_loader)      

# log.write('Test Accuracy:%.2f\n'%(acc))
# log.flush()
