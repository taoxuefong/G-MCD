import numpy
import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt

import sys

sys.path.append("..")
sys.path.append("../../config/pycharm-debug-py3k.egg")
import models
import dataset
from utils import *
import evaluate
import loss
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step', default=50, type=int, metavar='N',
                    help=' period of learning rate decay.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='G',
                    help='set the ID of GPU')
parser.add_argument('-exp', '--exp-name', type=str, default='0000', help='the name of experiment')
parser.add_argument('--debug', action='store_true', help='use remote debug', default=False)

## data setting
parser.add_argument('--data', choices=dataset.__all__, help='dataset: ' +' | '.join(dataset.__all__) +
                        ' (default: MARKET)', default='MARKET')
parser.add_argument('--num', type=int, default=2, help='the total number of images for each sample')

parser.add_argument('--height', type=int, default=384)
parser.add_argument('--width', type=int, default=128)

## network
parser.add_argument('--net', type=str, default='PatchNetUn', choices=models.__all__, help='nets: ' +' | '.join(models.__all__) +
                        ' (default: PatchNet)')


## loss setting
parser.add_argument('--scale', type=float, default=15)
parser.add_argument('--mm', type=float, default=0.1, help=' the momentum of the memory')
parser.add_argument('--ploss', default=2, type=float, help='the weight of the PEDAL loss')
parser.add_argument('--iloss', default=1, type=float, help='the weight of the IFPL loss')
parser.add_argument('--margin', default=2, type=float, help='the margin of local consistence loss')
## pretrained model
parser.add_argument('--pre-name', default=None, type=str, help='use which pretrained model to initialize the model')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
net = models.__dict__[args.net]
sys.stdout = Logger(os.path.join('D:\\reid\code\PAUL-master', args.exp_name))
prec = {} # loss history
prec['rank1'] = []
prec['map'] = []
def main():

    global args
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    data = dataset.__dict__[args.data](part='train', size=(args.height, args.width), require_path=True, pseudo_pair=args.num)
    train_loader = torch.utils.data.DataLoader(
        data,
        shuffle=True, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)

    for i in range(13,100,1):
        pre_names=str(i) + '.pth'
        # create models
        if pre_names is not None:
            path = os.path.join('D:\\reid\code\PAUL-master\pytorch', pre_names)
            pretrained_model = torch.load(path)['state_dict']
            class_num = pretrained_model['module.new.fc_list.1.weight'].size(0)
            print('the class number of pretrained model is {}'.format(class_num))
        else:
            raise RuntimeError('require a pretrained model to initialize the weight')
        
        model = net(class_num=class_num)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(pretrained_model)
        # evaluate the directly transfer result
        save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
        }, exp_name=args.exp_name, is_best=False)
        
        extract()
        MAP,rank1=evaluate.eval_result(exp_name=args.exp_name, data=args.data)
        print('epoch{:<4} Mean AP: {:4.2%} rank-1: {:4.2%}'.format(i,MAP,rank1))
    
        prec['rank1'].append(rank1)
        prec['map'].append(MAP)
        draw_curve(i)
    

#绘制每个y预训练模型对精度和MAP的影响
x_epoch = []
fig = plt.figure()
def draw_curve(current_epoch):
    plt.title('rank-1 and MAP')  #标题
    x_epoch.append(current_epoch)
    plt.plot(x_epoch, prec['rank1'], 'bo-', label='rank1')
    plt.plot(x_epoch, prec['map'], 'ro-', label='MAP')
    if current_epoch == 0:
        plt.legend()
    fig.savefig( os.path.join('D:\\reid\code\PAUL-master','pre_train.jpg'))
def extract():

    model = net(is_for_test=True)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(os.path.join('D:\\reid\code\PAUL-master', args.exp_name, 'checkpoint.pth'))
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #print(args.start_epoch)
    # switch to evaluate mode
    model.eval()
    part = ['query', 'gallery']

    for p in part:
        val_loader = torch.utils.data.DataLoader(
            dataset.__dict__[args.data](part=p,  require_path=True, size=(args.height, args.width)),
            batch_size=args.batch_size*args.num, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        with torch.no_grad():
            paths = []
            for i, (input, target, path, cam) in enumerate(val_loader):

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                feat_list = model(input)

                feat = torch.cat(feat_list, dim=1)

                feature = feat.cpu()
                target = target.cpu()

                nd_label = target.numpy()
                nd_feature = feature.numpy()
                if i == 0:
                    all_feature = nd_feature
                    all_label = nd_label
                    all_cam = cam.numpy()
                else:
                    all_feature = numpy.vstack((all_feature, nd_feature))
                    all_label = numpy.concatenate((all_label, nd_label))
                    all_cam = numpy.concatenate((all_cam, cam.numpy()))
                paths.extend(path)
            all_label.shape = (all_label.size, 1)
            all_cam.shape = (all_cam.size, 1)
            print(all_feature.shape, all_label.shape, all_cam.shape)
            save_feature(p, args.exp_name, args.data, all_feature, all_label, paths, all_cam)

if __name__ == '__main__':
    #python -W ignore serch_pretrain.py --data MARKET --gpu 0,1 --pre-name checkpoint.pth --exp-name snapshot --batch-size 42 --scale 15 --lr 0.0001 
    if args.debug:
        remote_debug()
    main()
