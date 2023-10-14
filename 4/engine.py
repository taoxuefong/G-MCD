from __future__ import division, print_function, absolute_import
import time
import numpy as np
import os.path as osp
import datetime
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import pickle
import sys
import os
import scipy.io as sio
from scipy.spatial import distance
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torchreid import metrics
from torchreid.metrics import compute_distance_matrix

from torchreid.utils import (
    AverageMeter, re_ranking, save_checkpoint, visualize_ranked_results
)
from torchreid.losses import DeepSupervision
from torchreid.losses import PatchMemory
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer=None,
            scheduler=None,
            use_gpu=True,
            mmd_only=True,
    ):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.train_loader_t = self.datamanager.train_loader_t
        self.mmd_only = mmd_only
        self.patch_centers = PatchMemory(momentum=0.1)
        #self.centers = SmoothingForImage(momentum=0.1)

    def run(
            self,
            save_dir='log',
            max_epoch=0,
            start_epoch=0,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
            start_eval=0,
            eval_freq=-1,
            test_only=False,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            save_best_only=True,
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
            save_best_only (bool, optional): during training, save the best model on test set and last epoch.
                Default is True to save storage.
        """

        if visrank and not test_only:
            raise ValueError(
                'visrank can be set to True only if test_only=True'
            )

        if test_only:
            self.test(
                0,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)
        
        # -------------------------------------------------------------------------------------------------------------------- #
        print('initialize the centers')
        self.model.train()
        for batch_idx,(imgs_t, pids_t, camids_t, paths_t) in enumerate(self.train_loader_t):
            # measure data loading time
            #imgs_t, pids_t, camids_t, paths_t = self._parse_data_for_train(data_t)
            with torch.no_grad():
                imgs_t = imgs_t.cuda()
                # compute output计算输出
                outputs_t, features_t = self.model(imgs_t)
                #feat_list,embedding_list
                self.patch_centers.get_soft_label(paths_t, camids_t, outputs_t)
                #self.centers.get_soft_label(paths_t, camids_t, features_t)
        
        print('initialization done')
        
        
        time_start = time.time()
        print('=> Start training')
        
        rank1_best = 0

        for epoch in range(start_epoch, max_epoch):
            
            self.train(
                epoch,
                max_epoch,
                self.writer,
                self.patch_centers,
                #self.centers,
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers,
            )
            
            if (epoch + 1) >= start_eval \
                    and eval_freq > 0 \
                    and (epoch + 1) % eval_freq == 0 \
                    and (epoch + 1) != max_epoch \
                    or epoch > 50:
                rank1 = self.test(
                    epoch,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                if not save_best_only:
                    self._save_checkpoint(epoch, rank1, save_dir)
                # save best epoch on test set
                elif rank1 >= rank1_best:
                    rank1_best = rank1
                    self._save_checkpoint(epoch, rank1, save_dir, is_best=True)
        if max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                epoch,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            # save last epoch anyway
            self._save_checkpoint(epoch, rank1, save_dir, is_best=False)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(self):
        r"""Performs training on source datasets for one epoch.

        This will be called every epoch in ``run()``, e.g.

        .. code-block:: python

            for epoch in range(start_epoch, max_epoch):
                self.train(some_arguments)

        .. note::

            This must be implemented in subclasses.
        """
        raise NotImplementedError

    def test(
            self,
            epoch,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            save_dir='',
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        targets = list(self.test_loader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = self.test_loader[name]['query']
            gallery_loader = self.test_loader[name]['gallery']
            rank1 = self._evaluate(
                epoch,
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )

        return rank1

    @torch.no_grad()
    def _evaluate(
            self,
            epoch,
            dataset_name='',
            query_loader=None,
            gallery_loader=None,
            dist_metric='euclidean',
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            save_dir='',
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False
    ):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self._parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self._extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.data.cpu()
                f_.append(features)
                #f_.append(features.numpy())
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            #f_ = np.concatenate(f_, axis=0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        query_feat = save_dir + '/query_features.mat'
        sio.savemat(query_feat,
                {"feat": qf,
                 "ids": q_pids,
                 "cam_ids": q_camids})
        '''
        with open(save_dir + '/query_features.pickle', 'wb') as f:
            pickle.dump([qf, q_pids], f)
        '''
        #print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        gallery_feat = save_dir + '/gallery_features.mat'
        sio.savemat(gallery_feat,
                {"feat": gf,
                 "ids": g_pids,
                 "cam_ids": g_camids})
        '''
        with open(save_dir + '/gallery_features.pickle', 'wb') as f:
            pickle.dump([gf, g_pids], f)
        '''
        #print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.
                    return_query_and_gallery_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0]

    def _compute_loss(self, criterion, outputs, targets):
        #if isinstance(outputs, (tuple, list)):
            #loss = DeepSupervision(criterion, outputs, camids, targets, camids_t)
        #else:
        loss = criterion(outputs, targets)
        return loss
    
    
    def _extract_features(self, input):
        self.model.eval()  # put in eval mode
        return self.model(input)
    ####################################
    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        paths = data[3]
        #imgids = data[4]
        return imgs, pids, camids, paths#, imgids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'epoch': epoch + 1,
                'rank1': rank1,
                #'optimizer': self.optimizer.state_dict(),
                #'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best
        )
