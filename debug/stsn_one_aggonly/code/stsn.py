# -*- coding: utf-8 -*- 
#@Author: lidong yu   
#@Date: 2020-01-16 14:36:42  
#@Last Modified by: lidong yu  
#@Last Modified time: 2020-01-16 14:36:42
#from IPython import embed
import torch.nn as nn
import logging
import torch
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import AGG,BACKBONES
from ..utils import build_conv_layer, build_norm_layer
import numpy as np
from mmcv.cnn import normal_init
from ..utils import ConvModule, bias_init_with_prob

@AGG.register_module
class STSN_one(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_one,self).__init__()
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18

        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        self.relu=nn.ReLU(inplace=False)
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels, offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=1, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels,  offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=1,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels, offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=1,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,2,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        # self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
        #                     padding=1,dilation=1,deformable_groups=1,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels, offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=1, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels,  offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=1,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,2,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        # self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
        #                     padding=1,dilation=1,deformable_groups=1,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels, offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=1, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels,  offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=1,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,2,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        # self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
        #                     padding=1,dilation=1,deformable_groups=1,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels, offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=1, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,2,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        # self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
        #                     padding=1,dilation=1,deformable_groups=1,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels, offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=1, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,2,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        # self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
        #                     padding=1,dilation=1,deformable_groups=1,bias=False)
        self.weight_feature1=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.weight_feature2=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.weight_feature3=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.weight_feature4=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.weight_feature5=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        
        
        self.offset=[]
        self.mask=[]
        print('init transform kernel')
        self.trans_kernel=torch.from_numpy(np.load('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy'))
        self.trans_kernel=nn.Parameter(self.trans_kernel)
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv11_offset, std=0.01)
        normal_init(self.conv11, std=0.01)
        normal_init(self.conv12_offset, std=0.01)
        normal_init(self.conv12, std=0.01)
        normal_init(self.conv13_offset, std=0.01)
        normal_init(self.conv13, std=0.01)
        normal_init(self.conv14_offset, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)

    def agg1(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset=self.conv14_offset(feature_f3)

            x = torch.linspace(-1, 1, reference.shape[-2])
            y = torch.linspace(-1, 1, reference.shape[-1])
            meshx, meshy = torch.meshgrid((x, y))
            grid = torch.stack((meshy, meshx), 2)
            grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            out = torch.nn.functional.grid_sample(support, grid)
            
            # print(torch.mean(reference-out))
            if test:
                return out,offset
            else:
                return out


    def agg2(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset=self.conv23_offset(feature_f2)
            
            x = torch.linspace(-1, 1, reference.shape[-2])
            y = torch.linspace(-1, 1, reference.shape[-1])
            meshx, meshy = torch.meshgrid((x, y))
            grid = torch.stack((meshy, meshx), 2)
            grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            out = torch.nn.functional.grid_sample(support, grid)
            
            # print(torch.mean(reference-out))
            
            if test:
                return out,offset
            else:
               return out
    def agg3(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset=self.conv33_offset(feature_f2)

            
            x = torch.linspace(-1, 1, reference.shape[-2])
            y = torch.linspace(-1, 1, reference.shape[-1])
            meshx, meshy = torch.meshgrid((x, y))
            grid = torch.stack((meshy, meshx), 2)
            grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            out = torch.nn.functional.grid_sample(support, grid)
            

            if test:
                return out,offset
            else:
               return out
    def agg4(self,support,reference,test=False):
        
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            offset1=self.conv41_offset(feature_f0)

            feature_f1=self.conv41(feature_f0,offset1)

            offset=self.conv42_offset(feature_f1)

            x = torch.linspace(-1, 1, reference.shape[-2])
            y = torch.linspace(-1, 1, reference.shape[-1])
            meshx, meshy = torch.meshgrid((x, y))
            grid = torch.stack((meshy, meshx), 2)
            grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            out = torch.nn.functional.grid_sample(support, grid)
            
            # print(torch.mean(reference-out))
            
            if test:
                return out,offset
            else:
               return out
    def agg5(self,support,reference,test=False):
        
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            offset1=self.conv51_offset(feature_f0)

            feature_f1=self.conv51(feature_f0,offset1)

            offset=self.conv52_offset(feature_f1)

            x=torch.linspace(-1,1,reference.shape[-2])
            y=torch.linspace(-1,1,reference.shape[-1])
            grid_x, grid_y = torch.meshgrid(x, y)
            grid=torch.stack([grid_y,grid_x],dim=2)
            grid=grid.repeat(reference.shape[0],1,1,1).to(reference.device)
            grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            out=torch.nn.functional.grid_sample(support,grid,mode='bilinear',align_corners=True)
            
            if test:
                return out,offset
            else:
               return out
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.weight_feature=[self.weight_feature1,self.weight_feature2,self.weight_feature3,self.weight_feature4,self.weight_feature5]
        if datas[0].device=='cuda:0':
            print('stsn one')
        refer_out=[]
        agg_out=[]
        support_out=[]
        support_count=3
        select_id=[]
        for i in range(support_count):
            support_out.append([])
            select_id.append([])
        out=[]
        select_id[0]=torch.arange(datas[0].shape[0])-2
        select_id[0]=torch.where(select_id[0]<0,torch.arange(datas[0].shape[0]),select_id[0])
        select_id[1]=torch.arange(datas[0].shape[0])+2
        select_id[1]=torch.where(select_id[1]>=datas[0].shape[0],torch.arange(datas[0].shape[0]),select_id[1])
        select_id[2]=torch.arange(datas[0].shape[0])
        for i in range(len(datas)):
            reference=datas[i]+0
            refer_out.append(reference)
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            # print(i,self.weight_feature[i])
            refer_weight_f=self.weight_feature[i](reference)
            feature=reference.unsqueeze(1)
            for j in range(support_count):
                support=datas[i][select_id[j],:,:,:]
                tk_feature=self.agg[i](support,reference)
                weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.weight_feature[i](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                support_out[j].append(tk_feature)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_out.append(agg_feature)
            if i==0 and weight.device==torch.ones(1).cuda(0).device:
                print('max_weight:',torch.max(weight).item(),'mean_max_weight:',torch.max(weight,dim=1)[0].mean().item())
        for j in range(support_count):
            support_out[j]=tuple(support_out[j])
        out=tuple(agg_out)

        return out
    def forward_pure(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.weight_feature=[self.weight_feature1,self.weight_feature2,self.weight_feature3,self.weight_feature4,self.weight_feature5]
        if datas[0].device=='cuda:0':
            print('stsn one')
        refer_out=[]
        agg_out=[]
        support_out=[]
        support_count=2
        select_id=[]
        for i in range(support_count):
            support_out.append([])
            select_id.append([])
        out=[]
        select_id=torch.arange(datas[0].shape[0])
        select_id=torch.where(select_id[0]<0,torch.arange(datas[0].shape[0]),select_id[0])
        support_warp=[]
        for i in range(len(datas)):
            reference=datas[i]+0
            refer_out.append(reference)
            feature=reference.unsqueeze(1)
            support=datas[i][select_id,:,:,:]
            tk_feature=self.agg[i](support,reference,test=False)
            
            support_warp.append(tk_feature)
        out=tuple(support_warp)

        return out
    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        self.offset=[]
        self.mask=[]
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0
            tk_feature,soffset4,mask4=self.agg[i](support,reference,test)
            self.offset.append(soffset4)
            self.mask.append(mask4)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)
    def forward_eval_pure(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.weight_feature=[self.weight_feature1,self.weight_feature2,self.weight_feature3,self.weight_feature4,self.weight_feature5]
        print('stsn eval')
        self.offset=[]
        self.mask=[]
        refer_out=[]
        agg_out=[]
        support_out=[]
        self.offset=[]
        self.mask=[]
        for i in range(datas[0].shape[0]-1):
            support_out.append([])
            self.offset.append([])
            self.mask.append([])
        out=[]
        
        support_count=2
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            refer_out.append(reference)
            support=datas[i][1:,:,:,:]+0
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            refer_weight_f=self.weight_feature[i](reference)
            feature=reference.unsqueeze(1)

            tk_feature,offset=self.agg[i](support[:1,...],reference,test)
            out.append(tk_feature)
        out=tuple(out)

        return out
    def forward_eval_back(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.weight_feature=[self.weight_feature1,self.weight_feature2,self.weight_feature3,self.weight_feature4,self.weight_feature5]
        print('stsn eval')
        refer_out=[]
        agg_out=[]
        support_out=[]
        self.offset=[]
        support_count=2
        for i in range(support_count):
            support_out.append([])
            self.offset.append([])
        out=[]

        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            refer_out.append(reference)
            support=datas[i][1:,:,:,:]+0
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            refer_weight_f=self.weight_feature[i](reference)
            feature=reference.unsqueeze(1)
            for j in range(support_count):
                tk_feature,offset=self.agg[i](support[j:j+1,...],reference,test)
                weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.weight_feature[i](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                support_out[j].append(tk_feature)
                self.offset[j].append(offset)
            weight=torch.nn.functional.softmax(weight0[:,1:,...],dim=1)
            agg_feature=torch.sum(feature[:,1:,...]*weight,dim=1)
            agg_out.append(agg_feature)
        for i in range(support_count):
            support_out[i]=tuple(support_out[i])
        out=[tuple(refer_out),tuple(agg_out)]+support_out

        return out
    def forward_eval(self,datas,test=False):
            # torch.Size([2, 256, 48, 156])
            # torch.Size([2, 256, 24, 78])
            # torch.Size([2, 256, 12, 39])
            # torch.Size([2, 256, 6, 20])
            # torch.Size([2, 256, 3, 10])
            output=[]
            self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
            self.weight_feature=[self.weight_feature1,self.weight_feature2,self.weight_feature3,self.weight_feature4,self.weight_feature5]
            if datas[0].device=='cuda:0':
                print('stsn one')
            refer_out=[]
            refer_trans_out=[]
            agg_out=[]
            agg_refer_out=[]
            support_out=[]
            support_count=2
            offsets=[]
            for i in range(support_count):
                support_out.append([])
                offsets.append([])
            out=[]
            for i in range(len(datas)):
                reference=datas[i][:1,...]+0
                refer_out.append(reference)
                refer_trans_out.append(self.agg[i](reference,reference))
                weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
                # print(i,self.weight_feature[i])
                refer_weight_f=self.weight_feature[i](reference)
                feature=reference.unsqueeze(1)
                for j in range(support_count):
                    support=datas[i][j+1:j+2,:,:,:]
                    tk_feature,offset=self.agg[i](support,reference,test=True)
                    weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.weight_feature[i](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
                    weight0=torch.cat([weight0,weight],dim=1)
                    feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                    support_out[j].append(tk_feature)
                    offsets[j].append(offset)
                
                weight=torch.nn.functional.softmax(weight0[:,1:,...],dim=1)
                agg_feature=torch.sum(feature[:,1:,...]*weight,dim=1)
                agg_out.append(agg_feature)
                weight=torch.nn.functional.softmax(weight0,dim=1)
                agg_feature=torch.sum(feature*weight,dim=1)
                agg_refer_out.append(agg_feature)
                # print(reference.shape,support.shape,agg_feature.shape)
                # exit()
                if i==0 and weight.device==torch.ones(1).cuda(0).device:
                    print('max_weight:',torch.max(weight).item(),'mean_max_weight:',torch.max(weight,dim=1)[0].mean().item())
            self.offset=offsets
            for j in range(support_count):
                support_out[j]=tuple(support_out[j])
            out=[tuple(refer_out),tuple(refer_trans_out),tuple(agg_out),tuple(agg_refer_out)]+support_out

            return out