from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid import metrics
from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.losses import MaximumMeanDiscrepancy
#from torchreid.losses import Minimum_Camera_Discrepancy
from torchreid.losses import PatchMemory
import torch
from functools import partial
from torch.autograd import Variable
from ..engine import Engine
from torchreid.metrics import compute_distance_matrix
import numpy as np
import pickle
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from torchreid.losses import TripletLoss, CrossEntropyLoss, Pedal, Ipfl


class ImageMmdEngine(Engine):

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=1,
            weight_x=1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True,
            mmd_only=True,
    ):
        super(ImageMmdEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu, mmd_only)

        self.optimizer.zero_grad()
        self.mmd_only = mmd_only ###
        self.weight_t = weight_t
        self.weight_x = weight_x
        '''
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        '''
        
        
        self.criterion_mmd = MaximumMeanDiscrepancy(
            instances=self.datamanager.train_loader.sampler.num_instances,
            batch_size=self.datamanager.train_loader.batch_size,
            global_only=False,
            distance_only=True,
            all=False
        )
        
        self.pc_criterion = Pedal(scale=15).cuda()
        self.pv_criterion = Ipfl(margin=5.0).cuda()
        '''
        self.criterion_mcd = Minimum_Camera_Discrepancy(
            instances=self.datamanager.train_loader.sampler.num_instances,
            batch_size=self.datamanager.train_loader.batch_size,
            global_only=False,
            distance_only=True,
            all=False
        )
        '''
    def train(
            self,
            epoch,
            max_epoch,
            writer,
            patch_centers,
            #centers,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
    ):
        #losses_triplet = AverageMeter()
        #losses_softmax = AverageMeter()
        #losses_p = AverageMeter()
        #losses_i = AverageMeter()
        losses_mmd_ww = AverageMeter()
        losses_mmd_wb = AverageMeter()
        #losses_mmd_bw = AverageMeter()
        losses_mmd_global = AverageMeter()
        #losses_mcds = AverageMeter()
        #losses_mmd_ww = AverageMeter()
        #losses_mmd_wb = AverageMeter()
        #losses_mmd_bw = AverageMeter()
        #losses_mmd_global = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader_t)
        end = time.time()
# -------------------------------------------------------------------------------------------------------------------- #
        for batch_idx, (data, data_t) in enumerate(zip(self.train_loader, self.train_loader_t)):
            data_time.update(time.time() - end)
            #########################################################
            imgs, pids, camids, paths = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                camids = camids.cuda()
            imgs_t, pids_t, camids_t, paths_t = self._parse_data_for_train(data_t)
            if self.use_gpu:
                imgs_t = imgs_t.cuda()
                camids_t = camids_t.cuda()
                #paths_t = paths_t.cuda()

            outputs, features = self.model(imgs)
            outputs_t, features_t = self.model(imgs_t)
            
            #loss_x = torch.sum(torch.stack([self._compute_loss(self.criterion_x, logits, logits, pids, pids) for logits in outputs], dim=0))
            all_feat = torch.cat(features, dim=1)
            #all_feat_t = torch.cat(features_t, dim=1)
            #loss_t = self._compute_loss(self.criterion_t, all_feat, all_feat, pids, pids)
            #loss_t = self._compute_loss(self.criterion_t, features, camids, pids, camids_t)
            #loss_x = self._compute_loss(self.criterion_x, outputs, camids, pids, camids_t)
            #loss_mcds = self._compute_loss(self.criterion_mcd, outputs, camids, features_t, camids_t)
            #loss = loss_t + loss_x #+ loss_mcds
            #######################################################################################################################################
            #feat = torch.stack(outputs_t, dim=0)
            #patch_agent, cam = patch_centers.get_soft_label(paths_t, camids_t, outputs_t)
            #ploss = self.pc_criterion(feat, camids_t, patch_agent, cam)
            all_embedding = torch.cat(features_t, dim=1)
            #iloss = self.pv_criterion(all_embedding, camids_t)
            #######################################################################################################################################
            #loss_mmd_ww, loss_mmd_wb, loss_mmd_bw, loss_mmd_global = self._compute_loss(self.criterion_mmd, all_feat, camids, all_feat_t, camids_t)
            #loss = loss_mmd_global + loss_mmd_ww + loss_mmd_wb + loss_mmd_bw + ploss
            #ploss = torch.tensor([0.]).cuda()
            all_embeddings = torch.cat(features, dim=1)
            '''
            loss_mmd_ww, loss_mmd_wb, loss_mmd_global = self.criterion_mmd(all_embeddings, camids, all_embedding, camids_t)
            #loss = ploss*2  + iloss + 10*(loss_mmd_ww + loss_mmd_wb + loss_mmd_bw + loss_mmd_global)
            loss = loss_mmd_ww + loss_mmd_wb + loss_mmd_global
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
# -------------------------------------------------------------------------------------------------------------------- #

            batch_time.update(time.time() - end)
            #losses_triplet.update(loss_t.item(), pids.size(0))
            #losses_softmax.update(loss_x.item(), pids.size(0))
            #losses_mcds.update(loss_mcds.item(), pids.size(0))
            losses_mmd_ww.update(loss_mmd_ww.item(), pids.size(0))
            losses_mmd_wb.update(loss_mmd_wb.item(), pids.size(0))
            #losses_mmd_bw.update(loss_mmd_bw.item(), pids.size(0))
            losses_mmd_global.update(loss_mmd_global.item(), pids.size(0))
            #losses_p.update(ploss.item(), pids.size(0))
            #losses_i.update(iloss.item(), pids.size(0))
            #losses_mmd_ww.update(loss_mmd_ww.item(), pids.size(0))
            #losses_mmd_wb.update(loss_mmd_wb.item(), pids.size(0))
            #losses_mmd_bw.update(loss_mmd_bw.item(), pids.size(0))
            #losses_mmd_global.update(loss_mmd_global.item(), pids.size(0))

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (
                        num_batches - (batch_idx + 1) + (max_epoch -
                                                         (epoch + 1)) * num_batches
                )
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                ##############################################################
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #'Loss_t {losses1.val:.4f} ({losses1.avg:.4f})\t'
                    #'Loss_x {losses2.val:.4f} ({losses2.avg:.4f})\t'
                    #'Loss_p {losses21.val:.4f} ({losses21.avg:.4f})\t'
                    #'Loss_i {losses22.val:.4f} ({losses22.avg:.4f})\t'
                    #'Loss_mcds {losses21.val:.4f} ({losses21.avg:.4f})\t'
                    'Loss_mmd_ww {losses3.val:.4f} ({losses3.avg:.4f})\t'
                    'Loss_mmd_wb {losses4.val:.4f} ({losses4.avg:.4f})\t'
                    #'Loss_mmd_bw {losses5.val:.4f} ({losses5.avg:.4f})\t'
                    'Loss_mmd_global {losses6.val:.4f} ({losses6.avg:.4f})\t'
                    #'Loss_mmd_wc {losses3.val:.4f} ({losses3.avg:.4f})\t'
                    #'Loss_mmd_bc {losses4.val:.4f} ({losses4.avg:.4f})\t'
                    #'Loss_mmd_global {losses5.val:.4f} ({losses5.avg:.4f})\t'
                    'Lr {lr:.6f}\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        #losses1=losses_triplet,
                        #losses2=losses_softmax,
                        #losses21=losses_p,
                        #losses22=losses_i,
                        losses3=losses_mmd_ww,
                        losses4=losses_mmd_wb,
                        #losses5=losses_mmd_bw,
                        losses6=losses_mmd_global,
                        #losses3=losses_mmd_ww,
                        #losses4=losses_mmd_bc,
                        #losses5=losses_mmd_global,
                        lr=self.optimizer.param_groups[0]['lr'],
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch * num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                #writer.add_scalar('Train/Loss_triplet', losses_triplet.avg, n_iter)
                #writer.add_scalar('Train/Loss_softmax', losses_softmax.avg, n_iter)
                #writer.add_scalar('Train/Loss_p', losses_p.avg, n_iter)
                #writer.add_scalar('Train/Loss_i', losses_i.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_ww', losses_mmd_ww.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_wb', losses_mmd_wb.avg, n_iter)
                #writer.add_scalar('Train/Loss_mmd_bw', losses_mmd_bw.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_global', losses_mmd_global.avg, n_iter)
                #writer.add_scalar('Train/Loss_mmd_bc', losses_mmd_bc.avg, n_iter)
                #writer.add_scalar('Train/Loss_mmd_wc', losses_mmd_wc.avg, n_iter)
                #writer.add_scalar('Train/Loss_mmd_global', losses_mmd_global.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )

            end = time.time()
            '''
        if self.scheduler is not None:
            self.scheduler.step()
            
        print_distri = True

        if print_distri:

            instances = self.datamanager.train_loader.sampler.num_instances
            batch_size = self.datamanager.train_loader.batch_size
            feature_size = 2048 # features_t.shape[1]  # 2048
            '''
            t = torch.reshape(features_t, (int(batch_size / instances), instances, feature_size))
            
            #  and compute bc/wc euclidean distance
            bct = compute_distance_matrix(t[0], t[0])
            wct = compute_distance_matrix(t[0], t[1])
            for i in t[1:]:
                bct = torch.cat((bct, compute_distance_matrix(i, i)))
                for j in t:
                    if j is not i:
                        wct = torch.cat((wct, compute_distance_matrix(i, j)))

            s = torch.reshape(features, (int(batch_size / instances), instances, feature_size))
            bcs = compute_distance_matrix(s[0], s[0])
            wcs = compute_distance_matrix(s[0], s[1])
            for i in s[1:]:
                bcs = torch.cat((bcs, compute_distance_matrix(i, i)))
                for j in s:
                    if j is not i:
                        wcs = torch.cat((wcs, compute_distance_matrix(i, j)))

            bcs = bcs.detach()
            wcs = wcs.detach()
            '''
            
            
            w_w_scamera, w_b_scamera, b_w_scamera, b_w_scamera1= [], [], [], []
            s = torch.reshape(all_feat, (int(batch_size / instances), instances, all_feat.shape[1]))
            #wws_camera, wbs_camera, 
            bws_camera = torch.tensor([]).cuda()
            for i in range(batch_size//instances):
                mask = camids[i*instances:(i+1)*instances].expand(instances, instances).eq(camids[i*instances:(i+1)*instances].expand(instances, instances).t())
                #print(mask)
                ws = compute_distance_matrix(s[i], s[i])
                for k in range(instances):
                    w_w_scamera.append(ws[k][mask[k]].unsqueeze(0))
                    w_b_scamera.append(ws[k][mask[k] == 0].unsqueeze(0))
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
            t = torch.reshape(all_feat_t, (int(batch_size / instances), instances, all_feat_t.shape[1]))
            #wws_camera, wbs_camera, bws_camera = [], [], []
            w_w_tcamera, w_b_tcamera, b_w_tcamera, b_w_tcamera1= [], [], [], []
            #wws_camera, wbs_camera, 
            bwt_camera = torch.tensor([]).cuda()
            for i in range(batch_size//instances):
                mask = camids[i*instances:(i+1)*instances].expand(instances, instances).eq(camids[i*instances:(i+1)*instances].expand(instances, instances).t())
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
                        mask1 = camids[i*instances:(i+1)*instances].expand(instances, instances).eq(camids[j*instances:(j+1)*instances].expand(instances, instances).t())
                        for k in range(instances):
                            if len(bt[k][mask1[k]])==0:
                                continue
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
                if len(b_w_tcamera)!=4:
                    continue
                
                if i==0:
                    bwt_camera = torch.cat(b_w_tcamera)
                else:
                    bw_camera = torch.cat(b_w_tcamera)
                    bwt_camera = torch.cat((bwt_camera,bw_camera))
                b_w_tcamera1.clear()
                b_w_tcamera.clear()
            '''
            bw_c = [x.cpu().detach().item() for x in bws_camera.flatten() if x > 0.00003]
            wb_c = [x.cpu().detach().item() for x in wbs_camera.flatten() if x > 0.00003]
            ww_c = [x.cpu().detach().item() for x in wws_camera.flatten() if x > 0.00003]
            data_wwc = norm.rvs(ww_c)
            sns.distplot(data_wwc, bins='auto', fit=norm, kde=False,color="#ff0000")
            
            data_wbc = norm.rvs(wb_c)
            sns.distplot(data_wbc, bins='auto', fit=norm, kde=False,color="#00ff00")
            
            data_bwc = norm.rvs(bw_c)
            sns.distplot(data_bwc, bins='auto', fit=norm, kde=False,color="#0000ff")
            
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequence of apparition')
            plt.title('Source Domain')
            plt.legend()
            plt.show()
            
            '''
            b_c = [x.cpu().detach().item() for x in bcs.flatten() if x > 0.00001]
            w_c = [x.cpu().detach().item() for x in wcs.flatten() if x > 0.00001]
            data_bc = norm.rvs(b_c)
            sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
            data_wc = norm.rvs(w_c)
            sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequence of apparition')
            plt.title('Source Domain')
            plt.legend()
            plt.show()

            b_ct = [x.cpu().detach().item() for x in bct.flatten() if x > 0.1]
            w_ct = [x.cpu().detach().item() for x in wct.flatten() if x > 0.1]
            data_bc = norm.rvs(b_ct)
            sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
            data_wc = norm.rvs(w_ct)
            sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequence of apparition')
            plt.title('Target Domain')
            plt.legend()
            plt.show()
            '''