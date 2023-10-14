from .resnet import *
from .resnet import Bottleneck, BasicBlock
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
from .eca_module import eca_layer
import random
import os
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
class PatchGenerator(nn.Module):
    def __init__(self):
        super(PatchGenerator, self).__init__()


        self.localization = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


        self.fc_loc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 2 * 3 * 6),
        )

        path_postion = [ 1, 0, 0, 0, 1/6, -5/6,
                          1, 0, 0, 0, 1/6, -3/6,
                          1, 0, 0, 0, 1/6, -1/6,
                          1, 0, 0, 0, 1/6, 1/6,
                          1, 0, 0, 0, 1/6, 3/6,
                          1, 0, 0, 0, 1/6, 5/6, ]
        
        #path_postion = [1, 5/6, 1/2, 1/3, 1/6, 4/6,
                        #1, 5/6, 1/2, 1/3, 1/6, 4/6,
                        #1, 5/6, 1/2, 1/3, 1/6, 4/6,
                        #1, 5/6, 1/2, 1/3, 1/6, 4/6,
                        #1, 5/6, 1/2, 1/3, 1/6, 4/6,
                        #1, 5/6, 1/2, 1/3, 1/6, 4/6, ]

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1,1))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 6, 2, 3)

        output = []
        for i in range(6):
            stripe = theta[:, i, :, :]
            grid = F.affine_grid(stripe, x.size())
            output.append(F.grid_sample(x, grid))

        return output


#获得初始化权重
class PatchNet(nn.Module):
    def __init__(self, class_num=1000, backbone=resnet50, pretrained=True, param_path='D:/reid/pretrain_model/resnet50-19c8e357.pth',
                 is_for_test=False):
        super(PatchNet,self).__init__()
        self.is_for_test = is_for_test
        self.backbone = backbone(pretrained=pretrained, param_path=param_path, remove=True, last_stride=1)
        #self.backbone = backbone(pretrained=pretrained, drop_last_stride=1)
        self.eca_layer=eca_layer(2048)
        self.bottleneck = nn.BatchNorm1d(256)
        self.bottleneck.bias.requires_grad_(False)
        #self.TripletAttention=TripletAttention()
        
        self.new = nn.ModuleList()
        self.stripe = 6
        down = nn.ModuleList([nn.Sequential(nn.Conv2d(2048, 256, 1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True)
                                            )
                              for _ in range(self.stripe)])

        self.new.add_module('down', down)
        if self.is_for_test is False:
            fc_list = nn.ModuleList([nn.Linear(256, class_num) for _ in range(self.stripe)])
            self.new.add_module('fc_list', fc_list)
            embedding = nn.ModuleList([nn.Linear(256, 128) for _ in range(self.stripe)])
            self.new.add_module('embedding', embedding)
		
        self._init_parameters()
        self.patch_proposal = PatchGenerator()

    def _init_parameters(self):
        for m in self.new.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.eca_layer(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        patch = self.patch_proposal(x)
		
        assert x.size(2) % self.stripe == 0

        local_feat_list = []
        logits_list = []
        embedding_list = []

        for i in range(self.stripe):

            local_feat = F.adaptive_avg_pool2d(patch[i], (1,1))
            #shape [N, c, 1, 1]

            local_feat = self.new.down[i](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            
            local_feat = self.bottleneck(local_feat)
            
            local_feat_list.append(local_feat)
            if self.is_for_test is False:

                logits_list.append(self.new.fc_list[i](local_feat))
                embedding_list.append(self.new.embedding[i](local_feat))

        if self.is_for_test is True:
            return local_feat_list
        
        return logits_list, local_feat_list, embedding_list

class PatchNetUn(PatchNet):
    def __init__(self, class_num=1000, backbone=resnet50, pretrained=False, is_for_test=False,
                 param_path='D:/reid/pretrain_model/resnet50-19c8e357.pth',spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8, height=384, width=128):
        super(PatchNetUn, self).__init__(class_num=class_num, backbone=backbone, param_path=param_path,
                                         pretrained=pretrained, is_for_test=is_for_test)
        self.eca_layer=eca_layer(2048)
        self.bottleneck = nn.BatchNorm1d(256)
        self.bottleneck.bias.requires_grad_(False)
        
        down = nn.ModuleList([nn.Sequential(nn.Conv2d(2048, 256, 1),
                                            nn.BatchNorm2d(256),
											#nn.ReLU(inplace=True)
											)
                              for _ in range(self.stripe)])

        self.new.add_module('down', down)
					
    def get_param(self, lr):

        return [{'params': self.new.down.parameters(), 'lr': lr},
                {'params': self.new.fc_list.parameters(), 'lr': lr},
                {'params': self.backbone.parameters(), 'lr': lr},
                {'params': self.patch_proposal.parameters(), 'lr': 0}]

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.eca_layer(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        patch = self.patch_proposal(x,)

        local_feat_list = []
        embedding_list = []

        for i in range(self.stripe):

            local_feat = F.adaptive_avg_pool2d(patch[i], (1,1))


            local_feat = self.new.down[i](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat = self.bottleneck(local_feat)
            local_feat = local_feat.renorm(2, 0, 1e-5).mul(1e5)
            local_feat_list.append(local_feat)
            if self.is_for_test is False:
                ew = self.new.embedding[i].weight
                eww = ew.renorm(2, 0, 1e-5).mul(1e5)
                esim = local_feat.mm(eww.t())
                embedding_list.append(esim)

        if self.is_for_test is True:
            return local_feat_list
        return local_feat_list, embedding_list
'''
if __name__ == '__main__':
    path = os.path.join('../../snapshot/checkpoint.pth')
    pretrained_model = torch.load(path)['state_dict']
    #加载该预训练模型
    class_num = pretrained_model['module.new.fc_list.1.weight'].size(0)
    model = PatchNetUn(class_num=class_num)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(pretrained_model)
    print(model)
    path_img = "D://reid/code/PAUL-master/1.jpg"
    img_pil = Image.open(path_img).convert('RGB')  # 0~255
    img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),#依据概率p对PIL图片进行水平翻转，默认0.5
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    input = img_transforms(img_pil)
    print(input.shape)
    output = model(input.reshape(1,3,384,128))
    print('net output size:')
    print(output.shape)
'''