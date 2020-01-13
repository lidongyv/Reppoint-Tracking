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
class STSN_trans(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_trans,self).__init__()

        offset_channels = 18

        #agg1
        self.offset1 = nn.Conv2d(in_channels,offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)

        self.sim1 = nn.Conv2d(in_channels,1,
                            kernel_size=1,stride=1,padding=0,dilation=1)
        #agg2
        self.offset2 = nn.Conv2d(in_channels,offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)

        self.sim2 = nn.Conv2d(in_channels,1,
                            kernel_size=1,stride=1,padding=0,dilation=1)
        #agg3
        self.offset3 = nn.Conv2d(in_channels,offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)

        self.sim3 = nn.Conv2d(in_channels,1,
                            kernel_size=1,stride=1,padding=0,dilation=1)
        #agg4
        self.offset4 = nn.Conv2d(in_channels,offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)

        self.sim4 = nn.Conv2d(in_channels,1,
                            kernel_size=1,stride=1,padding=0,dilation=1)

        #agg5
        self.offset5 = nn.Conv2d(in_channels,offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)

        self.sim5 = nn.Conv2d(in_channels,1,
                            kernel_size=1,stride=1,padding=0,dilation=1)

        self.grid_x=nn.Parameter(torch.arange(2000).view(2000,1).expand(2000,2000).float())
        self.grid_y=nn.Parameter(torch.arange(2000).view(1,2000).expand(2000,2000).float())
        self.relu=nn.LeakyReLU(inplace=True)
        self.offset=[]
        self.mask=[]
        # print('init transform kernel')
        # self.trans_kernel=torch.from_numpy(np.load('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy'))
        # self.trans_kernel=nn.Parameter(self.trans_kernel)
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
        normal_init(self.offset1, std=0.01)
        normal_init(self.sim1, std=0.01)
        normal_init(self.offset2, std=0.01)
        normal_init(self.sim2, std=0.01)
        normal_init(self.offset3, std=0.01)
        normal_init(self.sim3, std=0.01)
        normal_init(self.offset4, std=0.01)
        normal_init(self.sim4, std=0.01)
        normal_init(self.offset5, std=0.01)
        normal_init(self.sim5, std=0.01)
    def agg1(self,support,reference,test=False):
        n=reference.shape[0]
        h=reference.shape[-2]
        w=reference.shape[-1]
        fuse=torch.cat([support,reference],dim=1)
        offset=self.offset1(fuse)
        feature=[]
        weight=[]
        warp_grid=[]

        for i in range(9):
            grid_x=self.grid_x[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i,:,:]
            grid_y=self.grid_y[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i+1,:,:]
            grid=torch.cat([2*(grid_x.unsqueeze(3)/h-0.5),2*(grid_y.unsqueeze(3)/w-0.5)],dim=3)
            new_feature=torch.nn.functional.grid_sample(support,grid)
            feature.append(new_feature)
            weight.append(torch.nn.functional.cosine_similarity(reference,new_feature,dim=1).unsqueeze(1))
            warp_grid.append(grid)
        feature=torch.stack(feature,dim=4)
        weight=torch.stack(weight,dim=4)
        weight=torch.nn.functional.softmax(weight,dim=4)
        warp_feature=torch.sum(feature*weight,dim=4)
        # print(torch.stack(warp_grid,dim=4).shape,weight.shape)
        trans=torch.sum(torch.stack(warp_grid,dim=4)*weight.squeeze(1).unsqueeze(3),dim=4)
        
        if test:
            return warp_feature,trans,[offset,weight,trans]
        else:
            return warp_feature,trans
    def agg2(self,support,reference,test=False):
        n=reference.shape[0]
        h=reference.shape[-2]
        w=reference.shape[-1]
        fuse=torch.cat([support,reference],dim=1)
        offset=self.offset2(fuse)
        feature=[]
        weight=[]
        warp_grid=[]
        for i in range(9):
            grid_x=self.grid_x[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i,:,:]
            grid_y=self.grid_y[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i+1,:,:]
            grid=torch.cat([2*(grid_x.unsqueeze(3)/h-0.5),2*(grid_y.unsqueeze(3)/w-0.5)],dim=3)
            new_feature=torch.nn.functional.grid_sample(support,grid)
            feature.append(new_feature)
            weight.append(torch.nn.functional.cosine_similarity(reference,new_feature,dim=1).unsqueeze(1))
            warp_grid.append(grid)
        feature=torch.stack(feature,dim=4)
        weight=torch.stack(weight,dim=4)
        weight=torch.nn.functional.softmax(weight,dim=4)
        warp_feature=torch.sum(feature*weight,dim=4)
        # print(torch.stack(warp_grid,dim=4).shape,weight.shape)
        trans=torch.sum(torch.stack(warp_grid,dim=4)*weight.squeeze(1).unsqueeze(3),dim=4)
        
        if test:
            return warp_feature,trans,[offset,weight,trans]
        else:
            return warp_feature,trans
    def agg3(self,support,reference,test=False):
        n=reference.shape[0]
        h=reference.shape[-2]
        w=reference.shape[-1]
        fuse=torch.cat([support,reference],dim=1)
        offset=self.offset3(fuse)
        feature=[]
        weight=[]
        warp_grid=[]
        for i in range(9):
            grid_x=self.grid_x[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i,:,:]
            grid_y=self.grid_y[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i+1,:,:]
            grid=torch.cat([2*(grid_x.unsqueeze(3)/h-0.5),2*(grid_y.unsqueeze(3)/w-0.5)],dim=3)
            new_feature=torch.nn.functional.grid_sample(support,grid)
            feature.append(new_feature)
            weight.append(torch.nn.functional.cosine_similarity(reference,new_feature,dim=1).unsqueeze(1))
            warp_grid.append(grid)
        feature=torch.stack(feature,dim=4)
        weight=torch.stack(weight,dim=4)
        weight=torch.nn.functional.softmax(weight,dim=4)
        warp_feature=torch.sum(feature*weight,dim=4)
        # print(torch.stack(warp_grid,dim=4).shape,weight.shape)
        trans=torch.sum(torch.stack(warp_grid,dim=4)*weight.squeeze(1).unsqueeze(3),dim=4)
        
        if test:
            return warp_feature,trans,[offset,weight,trans]
        else:
            return warp_feature,trans
    def agg4(self,support,reference,test=False):
        n=reference.shape[0]
        h=reference.shape[-2]
        w=reference.shape[-1]
        fuse=torch.cat([support,reference],dim=1)
        offset=self.offset4(fuse)
        feature=[]
        weight=[]
        warp_grid=[]
        for i in range(9):
            grid_x=self.grid_x[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i,:,:]
            grid_y=self.grid_y[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i+1,:,:]
            grid=torch.cat([2*(grid_x.unsqueeze(3)/h-0.5),2*(grid_y.unsqueeze(3)/w-0.5)],dim=3)
            new_feature=torch.nn.functional.grid_sample(support,grid)
            feature.append(new_feature)
            weight.append(torch.nn.functional.cosine_similarity(reference,new_feature,dim=1).unsqueeze(1))
            warp_grid.append(grid)
        feature=torch.stack(feature,dim=4)
        weight=torch.stack(weight,dim=4)
        weight=torch.nn.functional.softmax(weight,dim=4)
        warp_feature=torch.sum(feature*weight,dim=4)
        # print(torch.stack(warp_grid,dim=4).shape,weight.shape)
        trans=torch.sum(torch.stack(warp_grid,dim=4)*weight.squeeze(1).unsqueeze(3),dim=4)
        
        if test:
            return warp_feature,trans,[offset,weight,trans]
        else:
            return warp_feature,trans

    def agg5(self,support,reference,test=False):

        n=reference.shape[0]
        h=reference.shape[-2]
        w=reference.shape[-1]
        fuse=torch.cat([support,reference],dim=1)
        offset=self.offset5(fuse)
        feature=[]
        weight=[]
        warp_grid=[]
        for i in range(9):
            grid_x=self.grid_x[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i,:,:]
            grid_y=self.grid_y[:h,:w].unsqueeze(0).expand(n,h,w)+offset[:,2*i+1,:,:]
            grid=torch.cat([2*(grid_x.unsqueeze(3)/h-0.5),2*(grid_y.unsqueeze(3)/w-0.5)],dim=3)
            new_feature=torch.nn.functional.grid_sample(support,grid)
            feature.append(new_feature)
            weight.append(torch.nn.functional.cosine_similarity(reference,new_feature,dim=1).unsqueeze(1))
            warp_grid.append(grid)
        feature=torch.stack(feature,dim=4)
        weight=torch.stack(weight,dim=4)
        weight=torch.nn.functional.softmax(weight,dim=4)
        warp_feature=torch.sum(feature*weight,dim=4)
        # print(torch.stack(warp_grid,dim=4).shape,weight.shape)
        trans=torch.sum(torch.stack(warp_grid,dim=4)*weight.squeeze(1).unsqueeze(3),dim=4)

        if test:
            return warp_feature,trans,[offset,weight,trans]
        else:
            return warp_feature,trans
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        print('shuffle id2:',shuffle_id2)
        for i in [-1,-2,-3,-4,-5]:
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            n=reference.shape[0]
            h=reference.shape[-2]
            w=reference.shape[-1]
            
            if i==-1:
                tk_feature,trans=self.agg[i](support,reference,test)
            else:
                trans=trans.transpose(1,3)
                trans=torch.nn.functional.interpolate(trans*2,(w,h)).transpose(1,3)
                support=torch.nn.functional.grid_sample(support,trans)
                tk_feature,trans=self.agg[i](support,reference,test)
            # weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            # weight0=torch.ones_like(weight1)
            # weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            # feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            # agg_feature=torch.sum(feature*weight,dim=1)
            output.append(tk_feature)
        return_out=[]
        for i in [-1,-2,-3,-4,-5]:
            return_out.append(output[i])
        return tuple(return_out)


    def forward_test(self,datas,test=True):
        output=[]
        self.offset=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]

        for i in [-1,-2,-3,-4,-5]:
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0
            # print(shuffle_id)
            # print(reference.shape,support.shape)
            n=reference.shape[0]
            h=reference.shape[-2]
            w=reference.shape[-1]
            if i==-1:
                tk_feature,trans,offset=self.agg[i](support,reference,test)
            else:
                trans=trans.transpose(1,3)
                trans=torch.nn.functional.interpolate(trans*2,(w,h)).transpose(1,3)
                support=torch.nn.functional.grid_sample(support,trans)
                tk_feature,trans,offset=self.agg[i](support,reference,test)
            # weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            # weight0=torch.ones_like(weight1)
            # weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            # feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            # agg_feature=torch.sum(feature*weight,dim=1)
            output.append(tk_feature)
            self.offset.append(offset)
        return_out=[]
        for i in [-1,-2,-3,-4,-5]:
            return_out.append(output[i])
        return tuple(return_out)
@AGG.register_module
class STSN(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv11_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv11(feature_f0, offset, mask)

            offset_mask2 = self.conv12_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)

            offset_mask3 = self.conv13_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)

            offset_mask4 = self.conv14_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            print(torch.max(mask,dim=1)[0].mean())
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset4=self.conv14_offset(feature_f3)
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(support,offset4)

        if test:
            return agg_features,offset4
        else:
            return agg_features


    def agg2(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv21_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(feature_f0, offset, mask)

            offset_mask2 = self.conv22_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)

            offset_mask4 = self.conv23_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset3=self.conv23_offset(feature_f2)
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg3(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv31_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(feature_f0, offset, mask)

            offset_mask2 = self.conv32_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)

            offset_mask4 = self.conv33_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset3=self.conv33_offset(feature_f2)
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg4(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv41_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv41(feature_f0, offset, mask)

                offset_mask4 = self.conv42_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv41_offset(feature_f0)

                feature_f1=self.conv41(feature_f0,offset1)

                offset2=self.conv42_offset(feature_f1)
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv51_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv51(feature_f0, offset, mask)

                offset_mask4 = self.conv52_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv51_offset(feature_f0)

                feature_f1=self.conv51(feature_f0,offset1)

                offset2=self.conv52_offset(feature_f1)
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        for i in range(len(datas)):
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            # print('split',support.device,reference.device,self.conv11.weight.device)
            tk_feature=self.agg[i](support,reference,test)
            output.append(tk_feature)
        return tuple(output)

    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0
            if self.with_modulated_dcn:
                tk_feature,soffset4,mask4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
                self.mask.append(mask4)
            else:
                tk_feature,soffset4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
            output.append(tk_feature)
        return tuple(output)
@AGG.register_module
class STSN_fuse(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_fuse,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv11_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv11(feature_f0, offset, mask)

            offset_mask2 = self.conv12_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)

            offset_mask3 = self.conv13_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)

            offset_mask4 = self.conv14_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            # print('mask weight',torch.max(mask,dim=1)[0].mean().item())
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset4=self.conv14_offset(feature_f3)
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(support,offset4)

        if test:
            return agg_features,offset4
        else:
            return agg_features


    def agg2(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv21_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(feature_f0, offset, mask)

            offset_mask2 = self.conv22_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)

            offset_mask4 = self.conv23_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset3=self.conv23_offset(feature_f2)
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg3(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv31_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(feature_f0, offset, mask)

            offset_mask2 = self.conv32_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)

            offset_mask4 = self.conv33_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset3=self.conv33_offset(feature_f2)
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg4(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv41_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv41(feature_f0, offset, mask)

                offset_mask4 = self.conv42_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv41_offset(feature_f0)

                feature_f1=self.conv41(feature_f0,offset1)

                offset2=self.conv42_offset(feature_f1)
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv51_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv51(feature_f0, offset, mask)

                offset_mask4 = self.conv52_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv51_offset(feature_f0)

                feature_f1=self.conv51(feature_f0,offset1)

                offset2=self.conv52_offset(feature_f1)
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        agg_output=[]
        refer_out=[]
        support1_out=[]
        support2_out=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        print('shuffle id2:',shuffle_id2)
        for i in range(len(datas)):
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            tk_feature1=self.agg[i](support.detach(),reference.detach(),test)
            support=datas[i][shuffle_id2,:,:,:]+0
            tk_feature2=self.agg[i](support.detach(),reference.detach(),test)
            weight1=torch.nn.functional.cosine_similarity(reference.detach(),tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight2=torch.nn.functional.cosine_similarity(reference.detach(),tk_feature2,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1,weight2],dim=1),dim=1)
            print('agg weight',(weight[:,0,...]).mean().item())
            feature=torch.cat([reference.unsqueeze(1),tk_feature1.unsqueeze(1),tk_feature2.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_output.append(agg_feature)
            refer_out.append(reference)
            support1_out.append(tk_feature1)
            support2_out.append(tk_feature2)
        # print(len(agg_output),len(refer_out),len(support1_out),print(support2_out))
        return [tuple(agg_output),tuple(refer_out),tuple(support1_out),tuple(support2_out)]

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

            if self.with_modulated_dcn:
                tk_feature,soffset4,mask4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
                self.mask.append(mask4)

            else:
                tk_feature,soffset4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)
    def forward_eval(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
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
        

        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            refer_out.append(reference)
            if datas[i].shape[0]>1:
                support=datas[i][1:,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][:1,:,:,:]+0
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            feature=reference.unsqueeze(1)
            for j in range(support.shape[0]):
                tk_feature,offset,mask=self.agg[i](support[j:j+1,...],reference,test)
                weight=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                support_out[j].append(tk_feature)
                self.offset[j].append(offset)
                self.mask[j].append(mask)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_out.append(agg_feature)
        for i in range(datas[0].shape[0]-1):
            support_out[i]=tuple(support_out[i])
        out=[tuple(refer_out),tuple(agg_out)]+support_out

        return out
@AGG.register_module
class STSN_fuse_t(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_fuse_t,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv11_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv11(feature_f0, offset, mask)

            offset_mask2 = self.conv12_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)

            offset_mask3 = self.conv13_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)

            offset_mask4 = self.conv14_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # print('mask weight',torch.max(mask,dim=1)[0].mean().item())
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset4=self.conv14_offset(feature_f3)
            # self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(support,offset4)

        if test:
            return agg_features,offset4
        else:
            return agg_features


    def agg2(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv21_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(feature_f0, offset, mask)

            offset_mask2 = self.conv22_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)

            offset_mask4 = self.conv23_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset3=self.conv23_offset(feature_f2)
            # self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg3(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv31_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(feature_f0, offset, mask)

            offset_mask2 = self.conv32_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)

            offset_mask4 = self.conv33_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset3=self.conv33_offset(feature_f2)
            # self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg4(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv41_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv41(feature_f0, offset, mask)

                offset_mask4 = self.conv42_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                # mask=torch.nn.functional.softmax(mask,dim=1)
                # kernel_weight=self.trans_kernel.detach()*9
                
                # self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv41_offset(feature_f0)

                feature_f1=self.conv41(feature_f0,offset1)

                offset2=self.conv42_offset(feature_f1)
                # self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv51_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv51(feature_f0, offset, mask)

                offset_mask4 = self.conv52_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                # mask=torch.nn.functional.softmax(mask,dim=1)
                # kernel_weight=self.trans_kernel.detach()*9
                
                # self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv51_offset(feature_f0)

                feature_f1=self.conv51(feature_f0,offset1)

                offset2=self.conv52_offset(feature_f1)
                # self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        agg_output=[]
        refer_out=[]
        support1_out=[]
        support2_out=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        # print('shuffle id:',shuffle_id)
        # print('shuffle id2:',shuffle_id2)
        for i in range(len(datas)):
            reference=datas[i]+0
            tk_feature0=self.agg[i](reference.detach(),reference.detach(),test)
            support=datas[i][shuffle_id,:,:,:]+0
            tk_feature1=self.agg[i](support.detach(),reference.detach(),test)
            support=datas[i][shuffle_id2,:,:,:]+0
            tk_feature2=self.agg[i](support.detach(),reference.detach(),test)

            weight1=torch.nn.functional.cosine_similarity(tk_feature0.detach(),tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight2=torch.nn.functional.cosine_similarity(tk_feature0.detach(),tk_feature2,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1,weight2],dim=1),dim=1)
            # print('agg weight',(weight[:,0,...]).mean().item())
            feature=torch.cat([tk_feature0.unsqueeze(1),tk_feature1.unsqueeze(1),tk_feature2.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_output.append(agg_feature)
            refer_out.append(tk_feature0)
            support1_out.append(tk_feature1)
            support2_out.append(tk_feature2)
        # print(len(agg_output),len(refer_out),len(support1_out),print(support2_out))
        return [tuple(agg_output),tuple(refer_out),tuple(support1_out),tuple(support2_out)]

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

            if self.with_modulated_dcn:
                tk_feature,soffset4,mask4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
                self.mask.append(mask4)

            else:
                tk_feature,soffset4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)
    def forward_eval(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
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
        

        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            
            if datas[i].shape[0]>1:
                support=datas[i][1:,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][:1,:,:,:]+0
            tk_feature0=self.agg[i](reference.detach(),reference.detach(),test)
            refer_out.append(tk_feature0)
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(tk_feature0,tk_feature0,dim=1).unsqueeze(1).unsqueeze(1))
            feature=tk_feature0.unsqueeze(1)
            for j in range(support.shape[0]):
                tk_feature,offset,mask=self.agg[i](support[j:j+1,...],reference,test)
                weight=torch.nn.functional.cosine_similarity(tk_feature0,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
                weight0=torch.cat([weight0,weight],dim=1)
                feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
                support_out[j].append(tk_feature)
                self.offset[j].append(offset)
                self.mask[j].append(mask)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_out.append(agg_feature)
        for i in range(datas[0].shape[0]-1):
            support_out[i]=tuple(support_out[i])
        out=[tuple(refer_out),tuple(agg_out)]+support_out

        return out
@AGG.register_module
class STSN_fuse_r(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_fuse_r,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
        self.offset=[]
        self.mask=[]
        print('init transform kernel')
        # for i in range(256):
        #     for j in range(256):
        #         for m in range(3):
        #             for n in range(3):
        #                 if i==j:
        #                     self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]/self.trans_kernel[i,j,m,n]
        #                     self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]/9
        #                 else:
        #                     self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]*0
        # np.save('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy',self.trans_kernel.data.cpu().numpy())

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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv11_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv11.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv11(feature_f0, offset, mask)

            offset_mask2 = self.conv12_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)

            offset_mask3 = self.conv13_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)

            offset_mask4 = self.conv14_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset4=self.conv14_offset(feature_f3)
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(support,offset4)

        if test:
            return agg_features,offset4
        else:
            return agg_features


    def agg2(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv21_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(feature_f0, offset, mask)

            offset_mask2 = self.conv22_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)

            offset_mask4 = self.conv23_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset3=self.conv23_offset(feature_f2)
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg3(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv31_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(feature_f0, offset, mask)

            offset_mask2 = self.conv32_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)

            offset_mask4 = self.conv33_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset3=self.conv33_offset(feature_f2)
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg4(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv41_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv41(feature_f0, offset, mask)

                offset_mask4 = self.conv42_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv41_offset(feature_f0)

                feature_f1=self.conv41(feature_f0,offset1)

                offset2=self.conv42_offset(feature_f1)
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv51_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv51(feature_f0, offset, mask)

                offset_mask4 = self.conv52_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv51_offset(feature_f0)

                feature_f1=self.conv51(feature_f0,offset1)

                offset2=self.conv52_offset(feature_f1)
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        print('shuffle id2:',shuffle_id2)
        for i in range(len(datas)):
            reference=datas[i]+0
            tk_feature0=self.agg[i](reference,reference,test)
            support=datas[i][shuffle_id,:,:,:]+0
            tk_feature1=self.agg[i](support,reference,test)
            support=datas[i][shuffle_id2,:,:,:]+0
            tk_feature2=self.agg[i](support,reference,test)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight2=torch.nn.functional.cosine_similarity(reference,tk_feature2,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.nn.functional.cosine_similarity(reference,tk_feature0,dim=1).unsqueeze(1).unsqueeze(1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1,weight2],dim=1),dim=1)
            feature=torch.cat([tk_feature0.unsqueeze(1),tk_feature1.unsqueeze(1),tk_feature2.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
            self.loss_trans=(torch.nn.functional.l1_loss(tk_feature0,reference.detach(),reduction='mean')+ \
                            0.1*torch.nn.functional.l1_loss(tk_feature1,reference.detach(),reduction='mean')+ \
                            0.1*torch.nn.functional.l1_loss(tk_feature2,reference.detach(),reduction='mean'))/1.2+ \
                        self.loss_trans
            self.loss_trans=self.loss_trans/len(datas)
        return tuple(output),self.loss_trans


    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        self.offset=[]
        for i in range(len(datas)):
            
            reference=datas[i][:1,:,:,:]+0
            tk_feature0,offset=self.agg[i](reference,reference,test)
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0

            if self.with_modulated_dcn:
                print(i)
                tk_feature1,offset_mask=self.agg[i](support,reference,test)
                self.offset.append(offset_mask)
            else:
               
                tk_feature1,offset=self.agg[i](support,reference,test)
                self.offset.append(offset)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.nn.functional.cosine_similarity(reference,tk_feature0,dim=1).unsqueeze(1).unsqueeze(1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([tk_feature0.unsqueeze(1),tk_feature1.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)
class STSN_fuse_ori(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_fuse_ori,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv11_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv11(feature_f0, offset, mask)

            offset_mask2 = self.conv12_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)

            offset_mask3 = self.conv13_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)

            offset_mask4 = self.conv14_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            offset1=self.conv11_offset(feature_f0)

            feature_f1=self.conv11(feature_f0,offset1)

            offset2=self.conv12_offset(feature_f1)

            feature_f2=self.conv12(feature_f1,offset2)

            offset3=self.conv13_offset(feature_f2)

            feature_f3=self.conv13(feature_f2,offset3)

            offset4=self.conv14_offset(feature_f3)
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(support,offset4)

        if test:
            return agg_features,offset4
        else:
            return agg_features


    def agg2(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv21_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(feature_f0, offset, mask)

            offset_mask2 = self.conv22_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)

            offset_mask4 = self.conv23_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv21_offset(feature_f0)

            feature_f1=self.conv21(feature_f0,offset1)

            offset2=self.conv22_offset(feature_f1)

            feature_f2=self.conv22(feature_f1,offset2)

            offset3=self.conv23_offset(feature_f2)
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg3(self,support,reference,test=False):

        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            offset_mask1 = self.conv31_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(feature_f0, offset, mask)

            offset_mask2 = self.conv32_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)

            offset_mask4 = self.conv33_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(support, offset, mask)
            if test:
                return out,offset,mask
            else:
                return out
        else:
            offset1=self.conv31_offset(feature_f0)

            feature_f1=self.conv31(feature_f0,offset1)

            offset2=self.conv32_offset(feature_f1)

            feature_f2=self.conv32(feature_f1,offset2)

            offset3=self.conv33_offset(feature_f2)
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(support,offset3)

        if test:
            return agg_features,offset3
        else:
            return agg_features

    def agg4(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv41_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv41(feature_f0, offset, mask)

                offset_mask4 = self.conv42_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv41_offset(feature_f0)

                feature_f1=self.conv41(feature_f0,offset1)

                offset2=self.conv42_offset(feature_f1)
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            feature_f0=torch.cat([support,reference],dim=1)

            if self.with_modulated_dcn:
                offset_mask1 = self.conv51_offset(feature_f0)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv51(feature_f0, offset, mask)

                offset_mask4 = self.conv52_offset(out)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(support, offset, mask)
                if test:
                    return out,offset,mask
                else:
                    return out
            else:
                offset1=self.conv51_offset(feature_f0)

                feature_f1=self.conv51(feature_f0,offset1)

                offset2=self.conv52_offset(feature_f1)
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(support,offset2)

            if test:
                return agg_features,offset2
            else:
                return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        print('shuffle id2:',shuffle_id2)
        for i in range(len(datas)):
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            tk_feature1=self.agg[i](support,reference,test)
            support=datas[i][shuffle_id2,:,:,:]+0
            tk_feature2=self.agg[i](support,reference,test)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight2=torch.nn.functional.cosine_similarity(reference,tk_feature2,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1,weight2],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature1.unsqueeze(1),tk_feature2.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)

    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0

            if self.with_modulated_dcn:
                tk_feature,soffset4,mask4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
                self.mask.append(mask4)

            else:
                tk_feature,soffset4=self.agg[i](support,reference,test)
                self.offset.append(soffset4)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)
@AGG.register_module
class STSN_ada_dcn(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_ada_dcn,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27

        self.conv1_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv1 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv2_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv3_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv3 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv4_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv4 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.relu=nn.LeakyReLU(inplace=True)
        self.neck=nn.Sequential(
            build_conv_layer(None, 256, 512, kernel_size=1, stride=1, padding=0,bias=False),
            nn.GroupNorm(32,512),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32,256),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 256, 256, kernel_size=1, stride=1, padding=0, bias=False))
        self.agg_set=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.offset=[]
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
    def agg1(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(features,offset1)
        offset2=self.conv2_offset(agg_features1)
        agg_features2=self.conv2(agg_features1,offset2)
        offset3=self.conv3_offset(agg_features2)
        agg_features3=self.conv3(agg_features2,offset3)
        offset4=self.conv4_offset(agg_features3)
        agg_features=self.conv4(support,offset4)
        if test:
            return agg_features,[offset1,offset2,offset3,offset4]
        else:
            return agg_features
    def agg2(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(features,offset1)
        offset2=self.conv2_offset(agg_features1)
        agg_features2=self.conv2(agg_features1,offset2)
        offset4=self.conv4_offset(agg_features2)
        agg_features=self.conv4(support,offset4)
        if test:
            return agg_features,[offset1,offset2,offset4]
        else:
            return agg_features
    def agg3(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(features,offset1)
        offset4=self.conv4_offset(agg_features1)
        agg_features=self.conv4(support,offset4)
        if test:
            return agg_features,[offset1,offset4]
        else:
            return agg_features
    def agg4(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset4=self.conv4_offset(features)
        agg_features=self.conv4(support,offset4)
        if test:
            return agg_features,[offset4]
        else:
            return agg_features
    def agg5(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset4=self.conv4_offset(features)
        agg_features=self.conv4(support,offset4)
        if test:
            return agg_features,[offset4]
        else:
            return agg_features

    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        for i in range(len(datas)):
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            tt_feature=self.agg_set[i](reference,reference,test)

            # print(self.roffset4.shape)
            stt=self.neck(tt_feature)
            # stt=tt_feature
            # ttweight=torch.exp(self.similarity(torch.cat([stt,stt],dim=1)).unsqueeze(1))#(b,1,w,h)
            ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)

            # print(ttweight.max(),ttweight.min())
            tk_feature=self.agg_set[i](support,reference,test)

            stk=self.neck(tk_feature)
            # stk=tk_feature
            # tkweight=torch.exp(self.similarity(torch.cat([stt,stk],dim=1)).unsqueeze(1))
            tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            # print(tkweight.max(),tkweight.min())
            weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            weights=F.softmax(weights,dim=0)
            print('support weight','scale:',i*8,torch.mean(weights[1,:,:,:]).item(),torch.min(weights[1,:,:,:]).item(),torch.max(weights[1,:,:,:]).item())
            features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(agg_features)
        # print(agg_features.shape)
        return tuple(output)
        # return stt
    def forward_test(self,datas,test=True):
        output=[]
        print('test')
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                # print(shuffle_id)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0
            
            # support=datas[:1,:,:,:]+0
            # print(datas.shape)
            tt_feature,roffset=self.agg_set[i](reference,reference,test)

            # print(self.roffset4.shape)
            stt=self.neck(tt_feature)
            # stt=tt_feature
            # ttweight=torch.exp(self.similarity(torch.cat([stt,stt],dim=1)).unsqueeze(1))#(b,1,w,h)
            ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)
            # print(ttweight.max(),ttweight.min())
            tk_feature,soffset=self.agg_set[i](support,reference,test)
            self.offset.append(soffset)
            stk=self.neck(tk_feature)
            # stk=tk_feature
            # tkweight=torch.exp(self.similarity(torch.cat([stt,stk],dim=1)).unsqueeze(1))
            tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            # print(tkweight.max(),tkweight.min())
            weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            weights=F.softmax(weights,dim=0)
            # print(torch.max((weights).abs()))
            self.weight=weights
            features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(agg_features)
            # print(agg_features.shape)
        return tuple(output)
@AGG.register_module
class STSN_atrous_dcn(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_atrous_dcn,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27

        self.conv1_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=2,dilation=2)
        self.conv1 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=2, dilation=2, deformable_groups=self.deformable_groups, bias=False)
        self.conv2_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv2 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=4,dilation=4,deformable_groups=self.deformable_groups,bias=False)
        self.conv3_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=8,dilation=8)
        self.conv3 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=8,dilation=8,deformable_groups=self.deformable_groups,bias=False)
        self.conv4_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=16,dilation=16)
        self.conv4 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=16,dilation=16,deformable_groups=self.deformable_groups,bias=False)
        self.relu=nn.LeakyReLU(inplace=True)
        self.neck=nn.Sequential(
            build_conv_layer(None, 256, 512, kernel_size=1, stride=1, padding=0,bias=False),
            nn.GroupNorm(32,512),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32,256),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 256, 256, kernel_size=1, stride=1, padding=0, bias=False))
        self.agg_set=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.offsets_record=[]
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
    def agg1(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(support,offset1)
        offset2=self.conv2_offset(features)
        agg_features2=self.conv2(support,offset2)
        offset3=self.conv3_offset(features)
        agg_features3=self.conv3(support,offset3)
        offset4=self.conv4_offset(features)
        agg_features4=self.conv4(support,offset4)
        agg_features=0.25*(agg_features1+agg_features2+agg_features3+agg_features4)
        if test:
            return agg_features,offset1,offset2,offset3,offset4
        else:
            return agg_features
    def agg2(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(support,offset1)
        offset2=self.conv2_offset(features)
        agg_features2=self.conv2(support,offset2)
        offset3=self.conv3_offset(features)
        agg_features3=self.conv3(support,offset3)
        agg_features=0.33*(agg_features1+agg_features2+agg_features3)
        if test:
            return agg_features,offset1,offset2,offset3
        else:
            return agg_features
    def agg3(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features1=self.conv1(support,offset1)
        offset2=self.conv2_offset(features)
        agg_features2=self.conv2(support,offset2)
        agg_features=0.5*(agg_features1+agg_features2)
        if test:
            return agg_features,offset1,offset2
        else:
            return agg_features
    def agg4(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features=self.conv1(support,offset1)
        if test:
            return agg_features,offset1
        else:
            return agg_features
    def agg5(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features=self.conv1(support,offset1)
        if test:
            return agg_features,offset1
        else:
            return agg_features

    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        for i in range(len(datas)):
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            tt_feature=self.agg_set[i](reference,reference,test)

            # print(self.roffset4.shape)
            stt=self.neck(tt_feature)
            # stt=tt_feature
            # ttweight=torch.exp(self.similarity(torch.cat([stt,stt],dim=1)).unsqueeze(1))#(b,1,w,h)
            ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)

            # print(ttweight.max(),ttweight.min())
            tk_feature=self.agg_set[i](support,reference,test)

            stk=self.neck(tk_feature)
            # stk=tk_feature
            # tkweight=torch.exp(self.similarity(torch.cat([stt,stk],dim=1)).unsqueeze(1))
            tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            # print(tkweight.max(),tkweight.min())
            weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            weights=F.softmax(weights,dim=0)
            print('support weight','scale:',i*8,torch.mean(weights[1,:,:,:]).item(),torch.min(weights[1,:,:,:]).item(),torch.max(weights[1,:,:,:]).item())
            features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(agg_features)
        # print(agg_features.shape)
        return tuple(output)
        # return stt
    def forward_test(self,datas,test=True):

        output=[]
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                # print(shuffle_id)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0
            
            # support=datas[:1,:,:,:]+0
            # print(datas.shape)
            tt_feature,roffset1,roffset2,roffset3,roffset4=self.agg_set[i](reference,reference,test)
            if(roffset1.shape[-1]==20):
                self.roffset1,self.roffset2,self.roffset3,self.roffset4=roffset1,roffset2,roffset3,roffset4
            # print(self.roffset4.shape)
            stt=self.neck(tt_feature)
            # stt=tt_feature
            # ttweight=torch.exp(self.similarity(torch.cat([stt,stt],dim=1)).unsqueeze(1))#(b,1,w,h)
            ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)
            # print(ttweight.max(),ttweight.min())
            tk_feature,soffset1,soffset2,soffset3,soffset4=self.agg_set[i](support,reference,test)
            if(soffset1.shape[-1]==20):
                self.soffset1,self.soffset2,self.soffset3,self.soffset4=soffset1,soffset2,soffset3,soffset4
            stk=self.neck(tk_feature)
            # stk=tk_feature
            # tkweight=torch.exp(self.similarity(torch.cat([stt,stk],dim=1)).unsqueeze(1))
            tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            # print(tkweight.max(),tkweight.min())
            weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            weights=F.softmax(weights,dim=0)
            # print(torch.max((weights).abs()))
            if(roffset1.shape[-1]==20):
                self.weight=weights
            features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(agg_features)
            # print(agg_features.shape)
        return tuple(output)
@AGG.register_module
class STSN_ms(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_ms,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        self.conv1_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv1 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv1_1 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv2_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv2_1 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv3_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv3 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv3_1 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv4_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv4 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv4_1 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv5_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv5 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.relu=nn.LeakyReLU(inplace=True)
        self.neck=nn.Sequential(
            build_conv_layer(None, 256, 512, kernel_size=1, stride=1, padding=0,bias=False),
            nn.GroupNorm(32,512),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32,256),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 256, 256, kernel_size=1, stride=1, padding=0, bias=False))
        self.offsets_record=[]
        self.agg_set=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.pre_dcn=[self.conv1_1,self.conv2_1,self.conv3_1,self.conv4_1]
        dcn_base = np.arange(-1,
                             2).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, 3)
        dcn_base_x = np.tile(dcn_base, 3)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
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
    def agg1(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv1_offset(features)
        agg_features=self.conv1(support,offset1)
        return agg_features,offset1
    def agg2(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv2_offset(features)
        agg_features=self.conv2(support,offset1)
        return agg_features,offset1
    def agg3(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv3_offset(features)
        agg_features=self.conv3(support,offset1)
        return agg_features,offset1
    def agg4(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv4_offset(features)
        agg_features=self.conv4(support,offset1)
        return agg_features,offset1
    def agg5(self,support,reference,test=False):
        features=torch.cat([support,reference],dim=1)
        offset1=self.conv5_offset(features)
        agg_features=self.conv5(support,offset1)
        return agg_features,offset1
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        offset_ori=self.dcn_base_offset
        for i in [4,3,2,1,0]:
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            bias=torch.zeros(reference.shape[0],2,reference.shape[-2],reference.shape[-1]).cuda(datas[0].device)
            template=torch.ones_like(bias).float()
            template[:,:,0]*=torch.reshape(torch.arange(template.shape[-2]).float(),(2000,1))
            template[:,:,1]*=torch.reshape(torch.arange(template.shape[-1]).float(),(1,2000))
            if i!=4:
                # offset_ori[:,:9,:,:]=self.dcn_base_offset[:,:9,:,:]*bias[:,0,:,:]*torch.mean(offset_ori[:,:9,:,:],dim=1,keepdim=True)
                # offset_ori[:,9:,:,:]=self.dcn_base_offset[:,9:,:,:]*bias[:,1,:,:]*torch.mean(offset_ori[:,9:,:,:],dim=1,keepdim=True)
                offset_ori=torch.nn.functional.interpolate(bias,scale_factor=2,mode='bilinear')*2
                # new_support=self.pre_dcn[i](support,offset_ori)
            tt_feature,_=self.agg_set[i](reference,reference,test)
            stt=self.neck(tt_feature)
            ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)
            tk_feature,offset=self.agg_set[i](new_support,reference,test)
            bias=offset.view(offset.shape[0],2,9,offset.shape[-2],offset.shape[-1])
            bias=torch.mean(bias, dim=2)
            stk=self.neck(tk_feature)
            tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            weights=F.softmax(weights,dim=0)
            print('support weight','scale:',i*8,torch.mean(weights[1,:,:,:]).item(),torch.min(weights[1,:,:,:]).item(),torch.max(weights[1,:,:,:]).item())
            features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(agg_features)
        # print(agg_features.shape)
        return tuple(output)
@AGG.register_module
class STSN_s2_ori(nn.Module):
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_s2_ori,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27

        self.conv1_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv1 = conv_op(in_channels, in_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv2_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv3_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv3 = conv_op(in_channels,in_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv4_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv4 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        # self.conv5 = DeformConv(out_channels,out_channels,kernel_size=3,stride=1,
        #                     padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.relu=nn.LeakyReLU(inplace=True)
        self.offset=[]
        # self.norm1 = nn.GroupNorm(32,in_channels)
        # self.norm2 = nn.GroupNorm(32,in_channels)
        # self.norm3 = nn.GroupNorm(32,in_channels)
        # self.norm4 = nn.GroupNorm(32,out_channels)
        # self.similarity=nn.Sequential(
        #     build_conv_layer(None, 512, 256, kernel_size=1, stride=1, padding=0,bias=False),
        #     nn.GroupNorm(32,256),
        #     nn.LeakyReLU(inplace=True),
        #     build_conv_layer(None, 256, 128, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.GroupNorm(32,128),
        #     nn.LeakyReLU(inplace=True),
        #     build_conv_layer(None, 128, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.GroupNorm(32,64),
        #     nn.LeakyReLU(inplace=True),
        #     build_conv_layer(None, 64, 1, kernel_size=1, stride=1, padding=0, bias=False),
        #     )
        self.neck=nn.Sequential(
            build_conv_layer(None, 256, 512, kernel_size=1, stride=1, padding=0,bias=False),
            nn.GroupNorm(32,512),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32,256),
            nn.LeakyReLU(inplace=True),
            build_conv_layer(None, 256, 256, kernel_size=1, stride=1, padding=0, bias=False))
        self.trans_kernel=self.conv4.weight.detach()
        print('init transform kernel')
        # for i in range(256):
        #     for j in range(256):
        #         for m in range(3):
        #             for n in range(3):
        #                 if i==j:
        #                     self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]/self.trans_kernel[i,j,m,n]
        #                     self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]/9
        #                 else:
        #                     self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]*0
        # np.save('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy',self.trans_kernel.data.cpu().numpy())
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
        normal_init(self.conv1_offset, std=0.01)
        normal_init(self.conv1, std=0.01)
        normal_init(self.conv2_offset, std=0.01)
        normal_init(self.conv2, std=0.01)
        normal_init(self.conv3_offset, std=0.01)
        normal_init(self.conv3, std=0.01)
        normal_init(self.conv4_offset, std=0.01)
        normal_init(self.conv4, std=0.01)

    def agg(self,support,reference,test=False):
        # features=torch.cat([support,reference],dim=1)
        feature_f0=torch.cat([support,reference],dim=1)
        print(feature_f0.device,self.conv1.weight.device)
        if self.with_modulated_dcn:
            offset_mask1 = self.conv1_offset(feature_f0)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv1(feature_f0, offset, mask)

            offset_mask2 = self.conv2_offset(out)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)

            offset_mask3 = self.conv3_offset(out)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv3(out, offset, mask)

            offset_mask4 = self.conv4_offset(out)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()

            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv4.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv4(support, offset, mask)
            if test:
                return out,offset_mask1,offset_mask2,offset_mask3,offset_mask4
            else:
                return out
        else:
            offset1=self.conv1_offset(feature_f0)

            feature_f1=self.conv1(feature_f0,offset1)

            offset2=self.conv2_offset(feature_f1)

            feature_f2=self.conv2(feature_f1,offset2)

            offset3=self.conv3_offset(feature_f2)

            feature_f3=self.conv3(feature_f2,offset3)

            offset4=self.conv4_offset(feature_f3)
            # print(self.conv4_offset.weight.shape)
            # print(self.conv4.weight.shape)
            
            self.conv4.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv4(support,offset4)
            self.offset4=offset4
            # offset4=torch.max(offset4.abs(),dim=1,keepdim=True).expand_as(offset4)
            # if offset4.shape[-1]==156:
            #     print('load')
            #     offset4_load=np.load('/home/ld/RepPoints/debug/agg_st_support/0/offset4.npy')
            #     offset4_load=torch.from_numpy(offset4_load).to(support.device)
            #     print('offset check in stsn',(offset4==offset4_load).all())
            #     #true for video 0
            #     support_load=np.load('/home/ld/RepPoints/debug/agg_st_support/2/support_f.npy')
            #     support_load=torch.from_numpy(support_load).to(support.device)
            #     print('support check in stsn',(support==support_load).all())
            #     agg_features=self.conv4(support_load,offset4_load)
            #     np.save('/home/ld/RepPoints/debug/feature_change/2/agg_f.npy',agg_features.data.cpu().numpy())
            #     np.save('/home/ld/RepPoints/debug/feature_change/2/support_f.npy',support_load.data.cpu().numpy())
            #     # np.save('/home/ld/RepPoints/debug/feature_change/0/refer_f.npy',reference.data.cpu().numpy())
            #     agg_load=np.load('/home/ld/RepPoints/debug/agg_st_support/2/agg_f.npy')
            #     agg_load=torch.from_numpy(agg_load).to(support.device)
            #     print('agg check in stsn',(agg_features==agg_load).all())
            #     # exit()
            #     #True for video 2
            #     self.offset4=offset4
            #     self.support_f=support
            #     self.agg_f=agg_features
            #     self.refer_f=reference
            #     return agg_features,offset1,offset2,offset3,offset4
            #     #?shape
            #     # offset4=torch.rand_like(offset4)
            # else:
            #     agg_features=self.conv4(support,offset4)
        # y_offset=[]
        # for i in range(9):
        #     y_offset.append(offset4[:,2*i+1,:,:])
        # y_offset=torch.stack(y_offset,dim=0)
        # print(torch.max(y_offset.abs()))

        # agg_features=(agg_features+support)/2
        # agg_features=self.norm4(agg_features)
        # agg_features=self.relu(agg_features)
        if test:
            return agg_features,offset1,offset2,offset3,offset4
        else:
            return agg_features

    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        for i in range(len(datas)):
            reference=datas[i]+0
            # print('support weight','scale:',reference.shape[-1])
            support=datas[i][shuffle_id,:,:,:]+0
            # tt_feature=self.agg(reference,reference,test)

            # # print(self.roffset4.shape)
            # stt=self.neck(tt_feature)
            # # stt=tt_feature
            # # ttweight=torch.exp(self.similarity(torch.cat([stt,stt],dim=1)).unsqueeze(1))#(b,1,w,h)
            # ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)

            # print(ttweight.max(),ttweight.min())
            if reference.shape[-1]==78:
                tk_feature=self.agg(support,reference,test)
            else:
                tk_feature=support
            # stk=self.neck(tk_feature)
            # # stk=tk_feature
            # # tkweight=torch.exp(self.similarity(torch.cat([stt,stk],dim=1)).unsqueeze(1))
            # tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            # # print(tkweight.max(),tkweight.min())
            # weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            # weights=F.softmax(weights,dim=0)
            # # print('support weight','scale:',i*8,torch.mean(weights[1,:,:,:]).item(),torch.min(weights[1,:,:,:]).item(),torch.max(weights[1,:,:,:]).item())
            # features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            # agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(tk_feature)
        # print(agg_features.shape)
        return tuple(output)
        # return stt
    def forward_test(self,datas,test=True):
        output=[]
        print('stsn test')
        for i in range(len(datas)):
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                # print(shuffle_id)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0

            # print((reference-support).abs().sum())
            # tt_feature,roffset1,roffset2,roffset3,roffset4=self.agg(reference,reference,test)
            # stt=self.neck(tt_feature)
            # ttweight=torch.nn.functional.cosine_similarity(stt,stt,dim=1).unsqueeze(1)
            if reference.shape[-1]==78:
                tk_feature,soffset1,soffset2,soffset3,soffset4=self.agg(support,reference,test)
                self.offset.append(soffset4)
            else:
                tk_feature=support
            # stk=self.neck(tk_feature)
            # tkweight=torch.nn.functional.cosine_similarity(stt,stk,dim=1).unsqueeze(1)
            # weights=torch.cat([ttweight.unsqueeze(0),tkweight.unsqueeze(0)],dim=0)#(2,b,1,w,h)
            # weights=F.softmax(weights,dim=0)
            # features=torch.cat([tt_feature.unsqueeze(0),tk_feature.unsqueeze(0)],dim=0)#(2,b,c,w,h)
            # agg_features=torch.sum(weights*features,dim=0)#(b,c,w,h)
            output.append(tk_feature)
            # print(agg_features.shape)
        return tuple(output)
@AGG.register_module
class STSN_c(nn.Module):
    #learnable dcn weight
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_c,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.loss_trans=0
        self.relu=nn.LeakyReLU(inplace=True)
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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        if self.with_modulated_dcn:
            fuse=torch.cat([support,reference],dim=1)
            offset_mask1 = self.conv11_offset(fuse)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv11.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv11(support, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask2 = self.conv12_offset(fuse)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv12.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv12(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask3 = self.conv13_offset(fuse)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv13.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv13(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask4 = self.conv14_offset(fuse)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(out, offset, mask)
            if test:
                return out,[offset_mask1,offset_mask2,offset_mask3,offset_mask4]
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            fuse=torch.cat([support,reference],dim=1)
            offset1=self.conv11_offset(fuse)

            feature_f1=self.conv11(support,offset1)
            fuse=torch.cat([feature_f1,reference],dim=1)
            offset2=self.conv12_offset(fuse)

            feature_f2=self.conv12(feature_f1,offset2)
            fuse=torch.cat([feature_f2,reference],dim=1)
            offset3=self.conv13_offset(fuse)

            feature_f3=self.conv13(feature_f2,offset3)
            fuse=torch.cat([feature_f3,reference],dim=1)
            offset4=self.conv14_offset(fuse)
            # self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(feature_f3,offset4)

            if test:
                return agg_features,[offset1,offset2,offset3,offset4]
            else:
                return agg_features


    def agg2(self,support,reference,test=False):

        

        if self.with_modulated_dcn:
            fuse=torch.cat([support,reference],dim=1)
            offset_mask1 = self.conv21_offset(fuse)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv21.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv21(support, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask2 = self.conv22_offset(fuse)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv22.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv22(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask4 = self.conv23_offset(fuse)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(out, offset, mask)
            if test:
                return out,[offset_mask1,offset_mask2,offset_mask4]
            else:
                return out
        else:
            fuse=torch.cat([support,reference],dim=1)
            offset1=self.conv21_offset(fuse)

            feature_f1=self.conv21(support,offset1)
            fuse=torch.cat([feature_f1,reference],dim=1)
            offset2=self.conv22_offset(fuse)

            feature_f2=self.conv22(feature_f1,offset2)
            fuse=torch.cat([feature_f2,reference],dim=1)
            offset3=self.conv23_offset(fuse)
            # self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(feature_f2,offset3)

            if test:
                return agg_features,[offset1,offset2,offset3]
            else:
                return agg_features

    def agg3(self,support,reference,test=False):

        

        if self.with_modulated_dcn:
            fuse=torch.cat([support,reference],dim=1)
            offset_mask1 = self.conv31_offset(fuse)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv31.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv31(support, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask2 = self.conv32_offset(fuse)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv32.weight=nn.Parameter(self.trans_kernel.detach())
            out = self.conv32(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask4 = self.conv33_offset(fuse)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            # mask = mask.sigmoid()
            mask=torch.nn.functional.softmax(mask,dim=1)
            kernel_weight=self.trans_kernel.detach()*9
            
            self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(out, offset, mask)
            if test:
                return out,[offset_mask1,offset_mask2,offset_mask4]
            else:
                return out
        else:
            fuse=torch.cat([support,reference],dim=1)
            offset1=self.conv31_offset(fuse)

            feature_f1=self.conv31(support,offset1)
            fuse=torch.cat([feature_f1,reference],dim=1)
            offset2=self.conv32_offset(fuse)

            feature_f2=self.conv32(feature_f1,offset2)
            fuse=torch.cat([feature_f2,reference],dim=1)
            offset3=self.conv33_offset(fuse)
            # self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(feature_f2,offset3)

            if test:
                return agg_features,[offset1,offset2,offset3]
            else:
                return agg_features

    def agg4(self,support,reference,test=False):
        
            

            if self.with_modulated_dcn:
                fuse=torch.cat([support,reference],dim=1)
                offset_mask1 = self.conv41_offset(fuse)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv41.weight=nn.Parameter(self.trans_kernel.detach())
                out = self.conv41(support, offset, mask)
                fuse=torch.cat([out,reference],dim=1)
                offset_mask4 = self.conv42_offset(fuse)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(out, offset, mask)
                if test:
                    return out,[offset_mask1,offset_mask4]
                else:
                    return out
            else:
                fuse=torch.cat([support,reference],dim=1)
                offset1=self.conv41_offset(fuse)

                feature_f1=self.conv41(support,offset1)
                fuse=torch.cat([feature_f1,reference],dim=1)
                offset2=self.conv42_offset(fuse)
                # self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(feature_f1,offset2)

            if test:
                return agg_features,[offset1,offset2]
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            

            if self.with_modulated_dcn:
                fuse=torch.cat([support,reference],dim=1)
                offset_mask1 = self.conv51_offset(fuse)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv51.weight=nn.Parameter(self.trans_kernel.detach())
                out = self.conv51(support, offset, mask)
                fuse=torch.cat([out,reference],dim=1)
                offset_mask4 = self.conv52_offset(fuse)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                # mask = mask.sigmoid()
                mask=torch.nn.functional.softmax(mask,dim=1)
                kernel_weight=self.trans_kernel.detach()*9
                
                self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(out, offset, mask)
                if test:
                    return out,[offset_mask1,offset_mask4]
                else:
                    return out
            else:
                fuse=torch.cat([support,reference],dim=1)
                offset1=self.conv51_offset(fuse)

                feature_f1=self.conv51(support,offset1)
                fuse=torch.cat([feature_f1,reference],dim=1)
                offset2=self.conv52_offset(fuse)
                # self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(feature_f1,offset2)

                if test:
                    return agg_features,[offset1,offset2]
                else:
                    return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        print('shuffle id2:',shuffle_id2)
        self.loss_trans=0
        for i in range(len(datas)):
            reference=datas[i]+0
            tk_feature0=self.agg[i](reference,reference,test)
            support=datas[i][shuffle_id,:,:,:]+0
            tk_feature1=self.agg[i](support,reference,test)
            support=datas[i][shuffle_id2,:,:,:]+0
            tk_feature2=self.agg[i](support,reference,test)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight2=torch.nn.functional.cosine_similarity(reference,tk_feature2,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.nn.functional.cosine_similarity(reference,tk_feature0,dim=1).unsqueeze(1).unsqueeze(1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1,weight2],dim=1),dim=1)
            feature=torch.cat([tk_feature0.unsqueeze(1),tk_feature1.unsqueeze(1),tk_feature2.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
            self.loss_trans=(torch.nn.functional.l1_loss(tk_feature0,reference.detach(),reduction='mean')+ \
                            0.1*torch.nn.functional.l1_loss(tk_feature1,reference.detach(),reduction='mean')+ \
                            0.1*torch.nn.functional.l1_loss(tk_feature2,reference.detach(),reduction='mean'))/1.2+ \
                        self.loss_trans
            self.loss_trans=self.loss_trans/len(datas)
        return tuple(output),self.loss_trans

    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        self.offset=[]
        for i in range(len(datas)):
            
            reference=datas[i][:1,:,:,:]+0
            tk_feature0,offset=self.agg[i](reference,reference,test)
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0

            if self.with_modulated_dcn:
                print(i)
                tk_feature1,offset_mask=self.agg[i](support,reference,test)
                self.offset.append(offset_mask)
            else:
               
                tk_feature1,offset=self.agg[i](support,reference,test)
                self.offset.append(offset)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.nn.functional.cosine_similarity(reference,tk_feature0,dim=1).unsqueeze(1).unsqueeze(1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([tk_feature0.unsqueeze(1),tk_feature1.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(tk_feature0)
        return tuple(output)
class STSN_c_ori(nn.Module):
    #learnable dcn weight
    def __init__(self,in_channels,out_channels,dcn):
        super(STSN_c_ori,self).__init__()
        self.deformable_groups = dcn.get('deformable_groups', 1)
        self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_modulated_dcn:
            conv_op = DeformConv
            offset_channels = 18
        else:
            conv_op = ModulatedDeformConv
            offset_channels = 27
        #agg1
        self.conv11_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv11 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv12_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv12 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv13_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv13 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv14_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv14 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg2
        self.conv21_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv21 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv22_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv22 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv23_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv23 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg3
        self.conv31_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv31 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv32_offset = nn.Conv2d(in_channels, self.deformable_groups * offset_channels,
                            kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv32 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        self.conv33_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv33 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)
        #agg4
        self.conv41_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv41 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv42_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv42 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)


        #agg5
        self.conv51_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                                    kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv51 = conv_op(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1, dilation=1, deformable_groups=self.deformable_groups, bias=False)
        self.conv52_offset = nn.Conv2d(in_channels,self.deformable_groups * offset_channels,
                            kernel_size=3,stride=1,padding=1,dilation=1)
        self.conv52 = conv_op(out_channels,out_channels,kernel_size=3,stride=1,
                            padding=1,dilation=1,deformable_groups=self.deformable_groups,bias=False)

        self.relu=nn.LeakyReLU(inplace=True)
        self.offset=[]
        self.mask=[]
        print('init transform kernel')
        for i in range(256):
            for j in range(256):
                for m in range(3):
                    for n in range(3):
                        if i==j:
                            self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]/self.trans_kernel[i,j,m,n]
                            self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]/9
                        else:
                            self.trans_kernel[i,j,m,n]=self.trans_kernel[i,j,m,n]*0
        np.save('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy',self.trans_kernel.data.cpu().numpy())
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
        normal_init(self.conv14, std=0.01)
        normal_init(self.conv21_offset, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22_offset, std=0.01)
        normal_init(self.conv22, std=0.01)
        normal_init(self.conv23_offset, std=0.01)
        normal_init(self.conv23, std=0.01)
        normal_init(self.conv31_offset, std=0.01)
        normal_init(self.conv31, std=0.01)
        normal_init(self.conv32_offset, std=0.01)
        normal_init(self.conv32, std=0.01)
        normal_init(self.conv33_offset, std=0.01)
        normal_init(self.conv33, std=0.01)
        normal_init(self.conv41_offset, std=0.01)
        normal_init(self.conv41, std=0.01)
        normal_init(self.conv42_offset, std=0.01)
        normal_init(self.conv42, std=0.01)
        normal_init(self.conv51_offset, std=0.01)
        normal_init(self.conv51, std=0.01)
        normal_init(self.conv52_offset, std=0.01)
        normal_init(self.conv52, std=0.01)

    def agg1(self,support,reference,test=False):

        if self.with_modulated_dcn:
            fuse=torch.cat([support,reference],dim=1)
            offset_mask1 = self.conv11_offset(fuse)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv11(support, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask2 = self.conv12_offset(fuse)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv12(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask3 = self.conv13_offset(fuse)
            offset = offset_mask3[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask3[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv13(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask4 = self.conv14_offset(fuse)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv14(out, offset, mask)
            if test:
                return out,[offset_mask1,offset_mask2,offset_mask3,offset_mask4]
            else:
                return out
        else:
            # print('agg1',feature_f0.device,self.conv11_offset.weight.device)
            fuse=torch.cat([support,reference],dim=1)
            offset1=self.conv11_offset(fuse)

            feature_f1=self.conv11(support,offset1)
            fuse=torch.cat([feature_f1,reference],dim=1)
            offset2=self.conv12_offset(fuse)

            feature_f2=self.conv12(feature_f1,offset2)
            fuse=torch.cat([feature_f2,reference],dim=1)
            offset3=self.conv13_offset(fuse)

            feature_f3=self.conv13(feature_f2,offset3)
            fuse=torch.cat([feature_f3,reference],dim=1)
            offset4=self.conv14_offset(fuse)
            # self.conv14.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv14(feature_f3,offset4)

            if test:
                return agg_features,[offset1,offset2,offset3,offset4]
            else:
                return agg_features


    def agg2(self,support,reference,test=False):

        

        if self.with_modulated_dcn:
            fuse=torch.cat([support,reference],dim=1)
            offset_mask1 = self.conv21_offset(fuse)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv21(support, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask2 = self.conv22_offset(fuse)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv22(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask4 = self.conv23_offset(fuse)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv23(out, offset, mask)
            if test:
                return out,[offset_mask1,offset_mask2,offset_mask4]
            else:
                return out
        else:
            fuse=torch.cat([support,reference],dim=1)
            offset1=self.conv21_offset(fuse)

            feature_f1=self.conv21(support,offset1)
            fuse=torch.cat([feature_f1,reference],dim=1)
            offset2=self.conv22_offset(fuse)

            feature_f2=self.conv22(feature_f1,offset2)
            fuse=torch.cat([feature_f2,reference],dim=1)
            offset3=self.conv23_offset(fuse)
            # self.conv23.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv23(feature_f2,offset3)

            if test:
                return agg_features,[offset1,offset2,offset3]
            else:
                return agg_features

    def agg3(self,support,reference,test=False):

        

        if self.with_modulated_dcn:
            fuse=torch.cat([support,reference],dim=1)
            offset_mask1 = self.conv31_offset(fuse)
            offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv31(support, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask2 = self.conv32_offset(fuse)
            offset = offset_mask2[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask2[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv32(out, offset, mask)
            fuse=torch.cat([out,reference],dim=1)
            offset_mask4 = self.conv33_offset(fuse)
            offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            # mask=torch.nn.functional.softmax(mask,dim=1)
            # kernel_weight=self.trans_kernel.detach()*9
            
            # self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            out = self.conv33(out, offset, mask)
            if test:
                return out,[offset_mask1,offset_mask2,offset_mask4]
            else:
                return out
        else:
            fuse=torch.cat([support,reference],dim=1)
            offset1=self.conv31_offset(fuse)

            feature_f1=self.conv31(support,offset1)
            fuse=torch.cat([feature_f1,reference],dim=1)
            offset2=self.conv32_offset(fuse)

            feature_f2=self.conv32(feature_f1,offset2)
            fuse=torch.cat([feature_f2,reference],dim=1)
            offset3=self.conv33_offset(fuse)
            # self.conv33.weight=nn.Parameter(self.trans_kernel.detach())

            agg_features=self.conv33(feature_f2,offset3)

            if test:
                return agg_features,[offset1,offset2,offset3]
            else:
                return agg_features

    def agg4(self,support,reference,test=False):
        
            

            if self.with_modulated_dcn:
                fuse=torch.cat([support,reference],dim=1)
                offset_mask1 = self.conv41_offset(fuse)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv41(support, offset, mask)
                fuse=torch.cat([out,reference],dim=1)
                offset_mask4 = self.conv42_offset(fuse)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                # mask=torch.nn.functional.softmax(mask,dim=1)
                # kernel_weight=self.trans_kernel.detach()*9
                
                # self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv42(out, offset, mask)
                if test:
                    return out,[offset_mask1,offset_mask4]
                else:
                    return out
            else:
                fuse=torch.cat([support,reference],dim=1)
                offset1=self.conv41_offset(fuse)

                feature_f1=self.conv41(support,offset1)
                fuse=torch.cat([feature_f1,reference],dim=1)
                offset2=self.conv42_offset(fuse)
                # self.conv42.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv42(feature_f1,offset2)

            if test:
                return agg_features,[offset1,offset2]
            else:
                return agg_features
    def agg5(self,support,reference,test=False):
        
            

            if self.with_modulated_dcn:
                fuse=torch.cat([support,reference],dim=1)
                offset_mask1 = self.conv51_offset(fuse)
                offset = offset_mask1[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask1[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv51(support, offset, mask)
                fuse=torch.cat([out,reference],dim=1)
                offset_mask4 = self.conv52_offset(fuse)
                offset = offset_mask4[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask4[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                # mask=torch.nn.functional.softmax(mask,dim=1)
                # kernel_weight=self.trans_kernel.detach()*9
                
                # self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                out = self.conv52(out, offset, mask)
                if test:
                    return out,[offset_mask1,offset_mask4]
                else:
                    return out
            else:
                fuse=torch.cat([support,reference],dim=1)
                offset1=self.conv51_offset(fuse)

                feature_f1=self.conv51(support,offset1)
                fuse=torch.cat([feature_f1,reference],dim=1)
                offset2=self.conv52_offset(fuse)
                # self.conv52.weight=nn.Parameter(self.trans_kernel.detach())

                agg_features=self.conv52(feature_f1,offset2)

                if test:
                    return agg_features,[offset1,offset2]
                else:
                    return agg_features
    def forward(self,datas,test=False):
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        shuffle_id=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]=shuffle_id[shuffle_id==np.arange(datas[0].shape[0])]-1
        shuffle_id2=np.random.randint(low=0,high=datas[0].shape[0],size=datas[0].shape[0])
        shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]=shuffle_id2[shuffle_id2==np.arange(datas[0].shape[0])]-1
        print('shuffle id:',shuffle_id)
        print('shuffle id2:',shuffle_id2)
        for i in range(len(datas)):
            reference=datas[i]+0
            support=datas[i][shuffle_id,:,:,:]+0
            tk_feature1=self.agg[i](support,reference,test)
            support=datas[i][shuffle_id2,:,:,:]+0
            tk_feature2=self.agg[i](support,reference,test)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature1,dim=1).unsqueeze(1).unsqueeze(1)
            weight2=torch.nn.functional.cosine_similarity(reference,tk_feature2,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1,weight2],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature1.unsqueeze(1),tk_feature2.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)

    def forward_test(self,datas,test=True):
        output=[]
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        print('stsn test')
        for i in range(len(datas)):
            
            reference=datas[i][:1,:,:,:]+0
            if datas[i].shape[0]>1:
                shuffle_id=np.random.randint(low=1,high=datas[i].shape[0],size=1)
                support=datas[i][shuffle_id,:,:,:]+0
            else:
                shuffle_id=[0]
                support=datas[i][shuffle_id,:,:,:]+0

            if self.with_modulated_dcn:
                print(i)
                tk_feature,offset_mask=self.agg[i](support,reference,test)
                self.offset.append(offset_mask)
            else:
               
                tk_feature,offset=self.agg[i](support,reference,test)
                self.offset.append(offset)
            weight1=torch.nn.functional.cosine_similarity(reference,tk_feature,dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.ones_like(weight1)
            weight=torch.nn.functional.softmax(torch.cat([weight0,weight1],dim=1),dim=1)
            feature=torch.cat([reference.unsqueeze(1),tk_feature.unsqueeze(1)],dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            output.append(agg_feature)
        return tuple(output)