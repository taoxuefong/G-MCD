from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import numpy as np  #
from scipy.spatial import distance  #
from scipy.stats import norm  #
import matplotlib.pyplot as plt  #
import seaborn as sns  #
import pickle  #
import torch  #
from sklearn.cluster import KMeans #
import random
from torchreid.metrics import compute_distance_matrix

from functools import partial
from torch.autograd import Variable


class MaximumMeanDiscrepancy(nn.Module):

    """
    Implementation of MMD :
    https://github.com/shafiqulislamsumon/HARTransferLearning/blob/master/maximum_mean_discrepancy.py
    """

    def __init__(self, use_gpu=True, batch_size=32, instances=4, global_only=False, distance_only=True, all=False):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.instances = instances
        self.global_only = global_only
        self.distance_only = distance_only
        self.all = all

    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
    #             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
    #
    # f_of_X: batch_size * k
    # f_of_Y: batch_size * k
    def mmd_linear(self, f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)

    def mmd_rbf_accelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i+1)%batch_size
            t1, t2 = s1+batch_size, s2+batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return loss / float(batch_size)

    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

    def pairwise_distance(self, x, y):

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)
        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_.cuda())
        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))
        return cost

    def mmd_loss(self, source, target):

        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
            )
        loss_value = self.maximum_mean_discrepancy(source, target, kernel=gaussian_kernel)
        loss_value = loss_value
        return loss_value
    
    def forward(self, source_features, camids, target_features, camids_t):

        # group each images of the same identity together
        instances = self.instances
        batch_size = self.batch_size
        feature_size = target_features.shape[1]  # 2048
        '''
        t = torch.reshape(target_features, (int(batch_size / instances), instances, feature_size))
        wct = compute_distance_matrix(t[0], t[0])
        bct = compute_distance_matrix(t[0], t[1])
        for i in t[1:]:
            wct = torch.cat((wct, compute_distance_matrix(i, i)))
            for j in t:
                if not torch.equal(i, j): # if j is not i:
                    bct = torch.cat((bct, compute_distance_matrix(i, j)))
        '''
        ################################################################
        #wws_camera, wbs_camera, bws_camera = [], [], []
        w_w_scamera, w_b_scamera, b_w_scamera, b_w_scamera1= [], [], [], []
        s = torch.reshape(source_features, (int(batch_size / instances), instances, feature_size))
        #wws_camera, wbs_camera, 
        bws_camera = torch.tensor([]).cuda()
        for i in range(batch_size//instances):
            mask = camids[i*instances:(i+1)*instances].expand(instances, instances).eq(camids[i*instances:(i+1)*instances].expand(instances, instances).t())
            #print(mask)
            ws = compute_distance_matrix(s[i], s[i])
            for k in range(instances):
                w_w_scamera.append(ws[k][mask[k]].unsqueeze(0))
                w_b_scamera.append(ws[k][mask[k] == 0].unsqueeze(0))
            #print(len(w_w_scamera),len(w_b_scamera))
            #print(ww_camera)
            if i==0:
                wws_camera = torch.cat(w_w_scamera)
                wbs_camera = torch.cat(w_b_scamera)
                #print(wws_camera)
                #print(wbs_camera)
            else:
                ww_camera = torch.cat(w_w_scamera)
                wb_camera = torch.cat(w_b_scamera)
                wws_camera = torch.cat((wws_camera,ww_camera))
                wbs_camera = torch.cat((wbs_camera,wb_camera))
            w_w_scamera.clear()
            w_b_scamera.clear()
            
            number=0
            for j in range(batch_size//instances):
                if i != j:
                    bs = compute_distance_matrix(s[i], s[j])
                    mask1 = camids[i*instances:(i+1)*instances].expand(instances, instances).eq(camids[j*instances:(j+1)*instances].expand(instances, instances).t())
                    for k in range(instances):
                        if len(bs[k][mask1[k]])==0:
                            continue
                        elif number < 4:
                            b_w_scamera.append(bs[k][mask1[k]].unsqueeze(0))
                            number+=1
                        '''
                        elif number == 0 or number % 4 !=0:
                            b_w_scamera1.append(bs[k][mask1[k]].unsqueeze(0))
                            number+=1
                        elif number == 4:
                            b_w_scamera = b_w_scamera1
                            b_w_scamera1.clear()
                            b_w_scamera1.append(bs[k][mask1[k]].unsqueeze(0))
                            number+=1
                        elif number>4 and number % 4==0:
                            b_w_scamera=[(b_w_scamera[l]+b_w_scamera1[l])/2 for l in range(0,len(b_w_scamera))]
                            b_w_scamera1.clear()
                            b_w_scamera1.append(bs[k][mask1[k]].unsqueeze(0))
                            number+=1
                        '''
            #print(len(b_w_scamera))
            if len(b_w_scamera)!=4:
                continue
            
            if i==0:
                bws_camera = torch.cat(b_w_scamera)
            else:
                bw_camera = torch.cat(b_w_scamera)
                bws_camera = torch.cat((bws_camera,bw_camera))
            b_w_scamera1.clear()
            b_w_scamera.clear()

        '''
        s = torch.reshape(source_features, (int(batch_size / instances), instances, feature_size))
        wcs = compute_distance_matrix(s[0], s[0])
        bcs = compute_distance_matrix(s[0], s[1])
        for i in s[1:]:
            wcs = torch.cat((wcs, compute_distance_matrix(i, i)))
            for j in s:
                if not torch.equal(i, j): # if j is not i:
                    bcs = torch.cat((bcs, compute_distance_matrix(i, j)))
        '''

        t = torch.reshape(target_features, (int(batch_size / instances), instances, feature_size))
        #wws_camera, wbs_camera, bws_camera = [], [], []
        w_w_tcamera, w_b_tcamera, b_w_tcamera, b_w_tcamera1= [], [], [], []
        #wws_camera, wbs_camera, 
        bwt_camera = torch.tensor([]).cuda()
        for i in range(batch_size//instances):
            mask = camids_t[i*instances:(i+1)*instances].expand(instances, instances).eq(camids_t[i*instances:(i+1)*instances].expand(instances, instances).t())
            #print(mask)
            wt = compute_distance_matrix(t[i], t[i])
            for k in range(instances):
                w_w_tcamera.append(wt[k][mask[k]].unsqueeze(0))
                w_b_tcamera.append(wt[k][mask[k] == 0].unsqueeze(0))
            #print(ww_camera)
            if i==0:
                wwt_camera = torch.cat(w_w_tcamera)
                wbt_camera = torch.cat(w_b_tcamera)
                #print(wws_camera)
                #print(wbs_camera)
            else:
                ww_camera = torch.cat(w_w_tcamera)
                wb_camera = torch.cat(w_b_tcamera)
                wwt_camera = torch.cat((wwt_camera,ww_camera))
                wbt_camera = torch.cat((wbt_camera,wb_camera))
            w_w_tcamera.clear()
            w_b_tcamera.clear()
            
            number=0
            for j in range(batch_size//instances):
                if i != j:
                    bt = compute_distance_matrix(t[i], t[j])
                    mask1 = camids_t[i*instances:(i+1)*instances].expand(instances, instances).eq(camids_t[j*instances:(j+1)*instances].expand(instances, instances).t())
                    for k in range(instances):
                        if len(bt[k][mask1[k]])==0:
                            continue
                        elif number < 4:
                            b_w_tcamera.append(bt[k][mask1[k]].unsqueeze(0))
                            number+=1
                        '''
                        elif number == 0 or number % 4 !=0:
                            b_w_tcamera1.append(bt[k][mask1[k]].unsqueeze(0))
                            number+=1
                        elif number == 4:
                            b_w_tcamera = b_w_tcamera1
                            b_w_tcamera1.clear()
                            b_w_tcamera1.append(bt[k][mask1[k]].unsqueeze(0))
                            number+=1
                        elif number>4 and number % 4==0:
                            b_w_tcamera=[(b_w_tcamera[l]+b_w_tcamera1[l])/2 for l in range(0,len(b_w_tcamera))]
                            b_w_tcamera1.clear()
                            b_w_tcamera1.append(bt[k][mask1[k]].unsqueeze(0))
                            number+=1 
                        '''
            if len(b_w_tcamera)!=4:
                continue
            
            if i==0:
                bwt_camera = torch.cat(b_w_tcamera)
            else:
                bw_camera = torch.cat(b_w_tcamera)
                bwt_camera = torch.cat((bwt_camera,bw_camera))
            b_w_tcamera1.clear()
            b_w_tcamera.clear()
            
            
        # We want to modify only target distribution
        '''
        bcs = bcs.detach()
        wcs = wcs.detach()
        return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct), self.mmd_loss(source_features, target_features)
        '''
        wws_camera = wws_camera.detach()
        wbs_camera = wbs_camera.detach()
        bws_camera = bws_camera.detach()
        
        return self.mmd_loss(wws_camera, wwt_camera), self.mmd_loss(wbs_camera, wbt_camera), self.mmd_loss(source_features, target_features)
        #return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct), torch.tensor(0)
        #return torch.tensor(0), self.mmd_loss(bcs, bct), torch.tensor(0)
        #return self.mmd_loss(wcs, wct), torch.tensor(0), torch.tensor(0)
