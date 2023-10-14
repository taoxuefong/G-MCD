# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:16:02 2021

@author: dell
"""
import torch
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable, Function
from scipy.stats import norm
import torch.nn.functional as F
import numpy as np
class Pedal(nn.Module):

    def __init__(self, scale=15, k=10):
        super(Pedal, self).__init__()
        self.scale =scale
        self.k = 10


    def forward(self, feature, camid, centers, cam):

        loss = 0
        #feature.size(0)=6
        #print(feature.size(),centers.size(),position.size())
        #6*96*512    6*(188->283->377)*512       96
        for p in range(feature.size(0)):
            part_feat = feature[p, :, :]
            part_centers = centers[p, :, :]
            m, n = part_feat.size(0), part_centers.size(0)
            dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                       part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_map.addmm_(1, -2, part_feat, part_centers.t())
            
            mask_intra = cam.expand(m, n).eq(camid.expand(n, m).t())
            
            for i in range(m):
                dist_map_intra = dist_map[i][mask_intra[i]]
                dist_map_inter = dist_map[i][mask_intra[i]==0]
                neg, _ = dist_map_intra.sort()
                neg2, _ = dist_map_inter.sort()
                #trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
                #neg, _ = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
                x_intra = ((-1 * self.scale * neg[:4]).exp().sum()).log()
                y_intra = ((-1 * self.scale * neg[6:]).exp().sum()).log()
                
                x_inter = ((-1 * self.scale * neg2[:8]).exp().sum()).log()
                y_inter = ((-1 * self.scale * neg2[12:]).exp().sum()).log()
                #print(y_inter-x_inter)
                dist_hinge_intra = torch.clamp(-x_intra + y_intra+ 12, min=0.0)
                dist_hinge_inter = torch.clamp(-x_inter + y_inter+ 5, min=0.0)
                #print(dist_hinge_intra, dist_hinge_inter)
                loss += (dist_hinge_intra + dist_hinge_inter)
        loss = loss.div(feature.size(1)).div(feature.size(0))
        return loss

class Ipfl(nn.Module):
    def __init__(self, margin=1.0, scale=10, p=2, eps=1e-5, max_iter=96, nearest=6, swap=False):

        super(Ipfl, self).__init__()
        self.margin = margin
        self.scale = scale
        self.p = p
        self.eps = eps
        self.swap = swap
        self.max_iter = max_iter
        self.nearest = nearest
        #self.neighbor_eps = 0.8
        #self.cam2uid = defaultdict(list)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    '''
    def compute_mask(self, size, img_ids, cam_ids, device):
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            self.cam2uid[cam].append(i)
        
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1

        return mask_instance, mask_intra, mask_inter
    
    def forward(self, sim, camid):
        label = torch.arange(sim.size(0))
        sim_exp = torch.exp(sim * self.scale)
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(0), label, camid, sim.device)
        
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        neighbor_mask_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = neighbor_mask_intra.sum(dim=1)

        sim_exp_intra = sim_exp * mask_intra
        score_intra = sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)
        #print(score_intra)
        score_intra = score_intra.clamp_min(1e-5)
        #print(score_intra)
        intra_loss = -score_intra.log().mul(neighbor_mask_intra).sum(dim=1).div(num_neighbor_intra).mean()
        #print(intra_loss)
        intra_loss -= score_intra.masked_select(mask_instance.bool()).log().mean()
        
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        neighbor_mask_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = neighbor_mask_inter.sum(dim=1)

        sim_exp_inter = mask_inter * sim_exp
        score_inter = sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True)
        score_inter = score_inter.clamp_min(1e-5)
        inter_loss = -score_inter.log().mul(neighbor_mask_inter).sum(dim=1).div(num_neighbor_inter).mean()
        '''
    def forward(self, feature, camid):
        loss_intra, loss_inter = 0, 0
        n = feature.size(0)
        dist_map = feature.pow(2).sum(dim=1, keepdim=True).expand(n, n) + \
                   feature.pow(2).sum(dim=1, keepdim=True).expand(n, n).t() + self.eps
        dist_map.addmm_(1, -2, feature, feature.t())
        dist_map = dist_map.clamp(min=1e-12).sqrt()
        mask = camid.expand(n, n).eq(camid.expand(n, n).t())
        #sorted, index = dist_map.sort(dim=1)嘿嘿
        #dist_intrap, dist_intran, dist_interp, dist_intern = [], [], [], []
        for i in range(n):
            '''
            dist_intrap.append(dist_map[i][mask[i]].min().unsqueeze(0))
            dist_intran.append(dist_map[i][mask[i]].max().unsqueeze(0))
            dist_interp.append(dist_map[i][mask[i] == 0].min().unsqueeze(0))
            dist_intern.append(dist_map[i][mask[i] == 0].max().unsqueeze(0))
            '''
            sorted1, index1 = dist_map[i][mask[i]].sort()
            sorted2, index2 = dist_map[i][mask[i]== 0].sort()
            #loss_intra += torch.clamp(sorted1[0:1].sum() -sorted1[4:].sum()/n, min=0.0)
            x_intra = ((-1 * self.scale * sorted1[:1]).exp().sum()).log()
            y_intra = ((-1 * self.scale * sorted1[4:]).exp().sum()).log()
            #print(-x_intra + y_intra)
            x_inter = ((-1 * self.scale * sorted2[:1]).exp().sum()).log()
            y_inter = ((-1 * self.scale * sorted2[6:]).exp().sum()).log()
            #print(-x_inter + y_inter)
            loss_intra += torch.clamp(-x_intra + y_intra +40, min=0.0)
            loss_inter += torch.clamp(-x_inter + y_inter+ 6, min=0.0)
            #print(loss_intra,loss_inter)
            #print(sorted)
            '''
            dist_intrap.extend(sorted1[0:2].unsqueeze(0))
            #dist_intrap.append(sorted1[1].unsqueeze(0))
            #dist_intrap.append(sorted1[2].unsqueeze(0))
            dist_intran.extend(sorted1[4:6].unsqueeze(0))
            #dist_intran.append(sorted1[5].unsqueeze(0))
            
            dist_interp.extend(sorted2[0:2].unsqueeze(0))
            #dist_interp.append(sorted2[1].unsqueeze(0))
            dist_intern.extend(sorted2[6:8].unsqueeze(0))
            #dist_intern.append(sorted2[5].unsqueeze(0))
        dist_intrap = torch.cat(dist_intrap)
        dist_intran = torch.cat(dist_intran)
        dist_interp = torch.cat(dist_interp)
        dist_intern = torch.cat(dist_intern)
        y_intra = torch.ones_like(dist_intran)
        y_inter = torch.ones_like(dist_intern)
            #same = sorted[i, :][label[index[i, :]] == label[i]]
            #diff = sorted[i, :][label[index[i, :]] != label[i]]
            #dist_hinge = torch.clamp(self.margin + diff[:self.nearest].sum()/self.nearest - diff[self.nearest:self.nearest*4].sum()/(self.nearest*2), min=0.0)
            #loss += dist_hinge
        #loss = loss.div(feature.size(0))

        #loss = intra_loss + inter_loss*0.5
        '''
        loss = (loss_intra + loss_inter*0.5) / (feature.size(0))
        return loss
    
    
    
    
    
    
    
    
    
    