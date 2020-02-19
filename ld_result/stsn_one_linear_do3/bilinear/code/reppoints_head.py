from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (PointGenerator, multi_apply, multiclass_nms,
                        point_target)
from mmdet.ops import DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .linearization import *

@HEADS.register_module
class RepPointsHead(nn.Module):
    """RepPoint head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01):
        super(RepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        #False
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, \
            "The points number should be an odd square number."
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        # array([-1, -1, -1,  0, -1,  1,  0, -1,  0,  0,  0,  1,  1, -1,  1,  0,  1,
        #         1])
        self._init_layers()
        self._init_agg()
    def _init_agg(self):
        in_channels=512
        offset_channels=18
        conv_op = DeformConv
        self.with_modulated_dcn=False
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
        self.cls_weight1=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.cls_weight2=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.cls_weight3=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.cls_weight4=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        self.cls_weight5=nn.Sequential(nn.Conv2d(256,512,
                                    kernel_size=1, stride=1,padding=0),
                                    self.relu,
                                    nn.Conv2d(512,512,
                                    kernel_size=3, stride=1,padding=1),
                                    self.relu,
                                    nn.Conv2d(512,2048,
                                    kernel_size=1, stride=1,padding=0))
        
        # self.reg_weight1=nn.Sequential(nn.Conv2d(256,512,
        #                             kernel_size=1, stride=1,padding=0),
        #                             self.relu,
        #                             nn.Conv2d(512,512,
        #                             kernel_size=3, stride=1,padding=1),
        #                             self.relu,
        #                             nn.Conv2d(512,2048,
        #                             kernel_size=1, stride=1,padding=0))
        # self.reg_weight2=nn.Sequential(nn.Conv2d(256,512,
        #                             kernel_size=1, stride=1,padding=0),
        #                             self.relu,
        #                             nn.Conv2d(512,512,
        #                             kernel_size=3, stride=1,padding=1),
        #                             self.relu,
        #                             nn.Conv2d(512,2048,
        #                             kernel_size=1, stride=1,padding=0))
        # self.reg_weight3=nn.Sequential(nn.Conv2d(256,512,
        #                             kernel_size=1, stride=1,padding=0),
        #                             self.relu,
        #                             nn.Conv2d(512,512,
        #                             kernel_size=3, stride=1,padding=1),
        #                             self.relu,
        #                             nn.Conv2d(512,2048,
        #                             kernel_size=1, stride=1,padding=0))
        # self.reg_weight4=nn.Sequential(nn.Conv2d(256,512,
        #                             kernel_size=1, stride=1,padding=0),
        #                             self.relu,
        #                             nn.Conv2d(512,512,
        #                             kernel_size=3, stride=1,padding=1),
        #                             self.relu,
        #                             nn.Conv2d(512,2048,
        #                             kernel_size=1, stride=1,padding=0))
        # self.reg_weight5=nn.Sequential(nn.Conv2d(256,512,
        #                             kernel_size=1, stride=1,padding=0),
        #                             self.relu,
        #                             nn.Conv2d(512,512,
        #                             kernel_size=3, stride=1,padding=1),
        #                             self.relu,
        #                             nn.Conv2d(512,2048,
        #                             kernel_size=1, stride=1,padding=0))
        self.offset=[]
        self.mask=[]
        # print('init transform kernel')
        # self.trans_kernel=torch.from_numpy(np.load('/home/ld/RepPoints/mmdetection/mmdet/ops/dcn/init_kernel.npy'))
        # self.trans_kernel=nn.Parameter(self.trans_kernel)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)
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

            # x = torch.linspace(-1, 1, reference.shape[-2])
            # y = torch.linspace(-1, 1, reference.shape[-1])
            # meshx, meshy = torch.meshgrid((x, y))
            # grid = torch.stack((meshy, meshx), 2)
            # grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            # grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            # grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            # out = torch.nn.functional.grid_sample(support, grid)
            
            return offset

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
            return offset
            # x = torch.linspace(-1, 1, reference.shape[-2])
            # y = torch.linspace(-1, 1, reference.shape[-1])
            # meshx, meshy = torch.meshgrid((x, y))
            # grid = torch.stack((meshy, meshx), 2)
            # grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            # grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            # grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            # out = torch.nn.functional.grid_sample(support, grid)
            
            # # print(torch.mean(reference-out))
            
            # if test:
            #     return out,offset
            # else:
            #    return out
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
            return offset
            
            # x = torch.linspace(-1, 1, reference.shape[-2])
            # y = torch.linspace(-1, 1, reference.shape[-1])
            # meshx, meshy = torch.meshgrid((x, y))
            # grid = torch.stack((meshy, meshx), 2)
            # grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            # grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            # grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            # out = torch.nn.functional.grid_sample(support, grid)
            

            # if test:
            #     return out,offset
            # else:
            #    return out
    def agg4(self,support,reference,test=False):
        
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            offset1=self.conv41_offset(feature_f0)

            feature_f1=self.conv41(feature_f0,offset1)

            offset=self.conv42_offset(feature_f1)
            return offset
            # x = torch.linspace(-1, 1, reference.shape[-2])
            # y = torch.linspace(-1, 1, reference.shape[-1])
            # meshx, meshy = torch.meshgrid((x, y))
            # grid = torch.stack((meshy, meshx), 2)
            # grid = grid.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            # grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            # grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            # out = torch.nn.functional.grid_sample(support, grid)
            
            # # print(torch.mean(reference-out))
            
            # if test:
            #     return out,offset
            # else:
            #    return out
    def agg5(self,support,reference,test=False):
        
        feature_f0=torch.cat([support,reference],dim=1)

        if self.with_modulated_dcn:
            print('module')
        else:
            offset1=self.conv51_offset(feature_f0)

            feature_f1=self.conv51(feature_f0,offset1)

            offset=self.conv52_offset(feature_f1)
            return offset
            # x=torch.linspace(-1,1,reference.shape[-2])
            # y=torch.linspace(-1,1,reference.shape[-1])
            # grid_x, grid_y = torch.meshgrid(x, y)
            # grid=torch.stack([grid_y,grid_x],dim=2)
            # grid=grid.repeat(reference.shape[0],1,1,1).to(reference.device)
            # grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            # grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            # out=torch.nn.functional.grid_sample(support,grid,mode='bilinear',align_corners=True)
            
            # if test:
            #     return out,offset
            # else:
            #    return out
    def points2bbox(self, pts, y_first=True):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            # tensor([-0.0048,  0.0101]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the
            regressed bboxes and generate the grids on the bboxes.
        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward_single(self, x,index,test=False):
        self.agg=[self.agg1,self.agg2,self.agg3,self.agg4,self.agg5]
        self.cls_weight=[self.cls_weight1,self.cls_weight2,self.cls_weight3,self.cls_weight4,self.cls_weight5]
        # self.reg_weight=[self.reg_weight1,self.reg_weight2,self.reg_weight3,self.reg_weight4,self.reg_weight5]
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        # print(dcn_base_offset.view(-1))
        # exit()

        # print(x.shape)
        # print(index)
        support_count=2
        select_id=[]
        for i in range(support_count):
            select_id.append([])
        # select_id[0]=torch.arange(x.shape[0])-2
        # select_id[0]=torch.where(select_id[0]<0,torch.arange(x.shape[0]),select_id[0])
        # select_id[1]=torch.arange(x.shape[0])+2
        # select_id[1]=torch.where(select_id[1]>=x.shape[0],torch.arange(x.shape[0]),select_id[1])

        select_id[0]=np.arange(x.shape[0])-1
        select_id[0]=np.where(select_id[0]<0,np.arange(x.shape[0])+1,select_id[0])
        select_id[1]=np.random.randint(low=0,high=x.shape[0],size=x.shape[0])
        select_id[1][select_id[1]==np.arange(x.shape[0])]=select_id[1][select_id[1]==np.arange(x.shape[0])]-2
        step=(select_id[1]-np.arange(x.shape[0]))/(select_id[0]-np.arange(x.shape[0]))
        step=torch.from_numpy(step).to(x.device).view(x.shape[0],1,1,1).float()


        reference=x+0

        with torch.no_grad():
            x_linear = torch.linspace(-1, 1, reference.shape[-2])
            y_linear = torch.linspace(-1, 1, reference.shape[-1])
            meshx, meshy = torch.meshgrid((x_linear, y_linear))
            grid_init = torch.stack((meshy, meshx), 2)
            grid_init = grid_init.unsqueeze(0).repeat(reference.shape[0],1,1,1).cuda(reference.device)
            # grid[:,:,:,0]=grid[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            # grid[:,:,:,1]=grid[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            # out = torch.nn.functional.grid_sample(support, grid)

        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        # initialize reppoints
        pts_init_feature=self.relu(self.reppoints_pts_init_conv(pts_feat))
        pts_out_init = self.reppoints_pts_init_out(pts_init_feature)

        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        #control the grad between two loss,0.1 valid grad
        #certain postion
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        #to relative positioni
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        if test:
            self.reppoints.append(dcn_offset.data.cpu().numpy())
        cls_out_feature=self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset))
        # cls_out = self.reppoints_cls_out(cls_out_feature)        

        pts_refine_feature=self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset))
        pts_out_refine = self.reppoints_pts_refine_out(pts_refine_feature)
        #detach the init grad
        pts_out_refine = pts_out_refine + pts_out_init.detach()



        if not test:
            #init the offset
            reference=cls_out_feature+0
            refer_weight_f=self.cls_weight[index](reference)
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            feature=reference.unsqueeze(1)
            
            support=cls_out_feature[select_id[0],:,:,:]
            offset=self.agg[index](support,reference)
            grid_cls_init=grid_init+0
            grid_cls_init[:,:,:,0]=grid_cls_init[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid_cls_init[:,:,:,1]=grid_cls_init[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            tk_feature,grad = grid_sample(support,reference, grid_cls_init,mode='linearized')
            weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.cls_weight[index](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.cat([weight0,weight],dim=1)
            feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
            #plan B
            #inv offset to warp reference by tht init offset
            inv_offset=self.agg[index](reference,support)
            #forward backward check
            #add loss
            #
            
            #use the init offset, warp the reference by offset
            #warp the reference by the offset
            # print(step.shape,inv_offset.shape)
            # torch.Size([10]) torch.Size([10, 2, 52, 160])
            inv_offset_xy=(inv_offset+0)*step.float()
            grid_cls_inv=grid_init+0
            grid_cls_inv[:,:,:,0]=grid_cls_inv[:,:,:,0]+inv_offset_xy[:,1,:,:]/reference.shape[-1]
            grid_cls_inv[:,:,:,1]=grid_cls_inv[:,:,:,1]+inv_offset_xy[:,0,:,:]/reference.shape[-2]
            image_linearized,_=grid_sample(reference+0,reference+0, grid_cls_inv,mode='linearized')
            #warp the warped reference back
            offset_xy=(offset+0)*step.float()
            # offset_xy=-torch.nn.functional.grid_sample(offset_xy,grid_cls_inv)+0
            # offset_xy=-offset_xy[:,:,grid_cls_inv[...,1],grid_cls_inv[...,0]]
            support=cls_out_feature[select_id[1],:,:,:]
            offset=self.agg[index](support,image_linearized)
            grid_cls_init=grid_init
            grid_cls_init[:,:,:,0]=grid_cls_init[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]+offset_xy[:,1,:,:]/reference.shape[-1]
            grid_cls_init[:,:,:,1]=grid_cls_init[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]+offset_xy[:,0,:,:]/reference.shape[-2]
            # print(support.device,image_linearized.device,grid_cls_init.device)
            tk_feature,_ = grid_sample(support,image_linearized, grid_cls_init,mode='linearized')
            weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.cls_weight[index](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.cat([weight0,weight],dim=1)
            feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
            
            #fuse feature computation
            weight=torch.nn.functional.softmax(weight0[:,1:,...],dim=1)
            agg_feature=torch.sum(feature[:,1:,...]*weight,dim=1)
            agg_cls_out = self.reppoints_cls_out(agg_feature)
            return agg_cls_out, pts_out_init, pts_out_refine
        else:
            step=torch.ones(1).cuda(x.device).float()*5
            grid_init=grid_init[:1,...]
            temp_offset=[]
            temp_inv_offset=[]
            reference=cls_out_feature[:1,...]+0
            refer_weight_f=self.cls_weight[index](reference)
            weight0=torch.ones_like(torch.nn.functional.cosine_similarity(reference,reference,dim=1).unsqueeze(1).unsqueeze(1))
            feature=reference.unsqueeze(1)
            #init offset
            support=cls_out_feature[1:2,:,:,:]
            offset=self.agg[index](support,reference)
            temp_offset.append(offset.data.cpu().numpy())
            grid_cls_init=grid_init+0
            grid_cls_init[:,:,:,0]=grid_cls_init[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]
            grid_cls_init[:,:,:,1]=grid_cls_init[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]
            tk_feature,grad = grid_sample(support,reference, grid_cls_init,mode='linearized')
            weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.cls_weight[index](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.cat([weight0,weight],dim=1)
            feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
            #plan B
            #inv offset to warp reference by tht init offset
            inv_offset=self.agg[index](reference,support)
            temp_inv_offset.append(inv_offset.data.cpu().numpy())
            
            # print(step.shape,inv_offset.shape)
            # torch.Size([10]) torch.Size([10, 2, 52, 160])
            inv_offset_xy=(inv_offset+0)*step.float()
            temp_inv_offset.append(inv_offset_xy.data.cpu().numpy())
            grid_cls_inv=grid_init+0
            # print(grid_cls_inv.shape,inv_offset_xy.shape)
            grid_cls_inv[:,:,:,0]=grid_cls_inv[:,:,:,0]+inv_offset_xy[:,1,:,:]/reference.shape[-1]
            grid_cls_inv[:,:,:,1]=grid_cls_inv[:,:,:,1]+inv_offset_xy[:,0,:,:]/reference.shape[-2]
            image_linearized,_=grid_sample(reference+0,reference+0, grid_cls_inv,mode='linearized')
            #warp the warped reference back
            offset_xy=(offset+0)*step.float()
            temp_offset.append(offset_xy.data.cpu().numpy())
            # offset_xy=-torch.nn.functional.grid_sample(offset_xy,grid_cls_inv)+0
            # offset_xy=-offset_xy[:,:,grid_cls_inv[...,1],grid_cls_inv[...,0]]
            support=cls_out_feature[2:3,:,:,:]
            offset=self.agg[index](support,image_linearized)
            temp_offset.append(offset.data.cpu().numpy())
            grid_cls_init=grid_init
            grid_cls_init[:,:,:,0]=grid_cls_init[:,:,:,0]+offset[:,1,:,:]/reference.shape[-1]+offset_xy[:,1,:,:]/reference.shape[-1]
            grid_cls_init[:,:,:,1]=grid_cls_init[:,:,:,1]+offset[:,0,:,:]/reference.shape[-2]+offset_xy[:,0,:,:]/reference.shape[-2]
            # print(support.device,image_linearized.device,grid_cls_init.device)
            tk_feature,_ = grid_sample(support,image_linearized, grid_cls_init,mode='linearized')
            weight=torch.nn.functional.cosine_similarity(refer_weight_f,self.cls_weight[index](tk_feature),dim=1).unsqueeze(1).unsqueeze(1)
            weight0=torch.cat([weight0,weight],dim=1)
            feature=torch.cat([feature,tk_feature.unsqueeze(1)],dim=1)
            weight=torch.nn.functional.softmax(weight0,dim=1)
            agg_feature=torch.sum(feature*weight,dim=1)
            agg_cls_out = self.reppoints_cls_out(agg_feature)
            self.offset.append(temp_offset)
            self.inv_offset.append(temp_inv_offset)
            # if len(self.offset)==1:
            #     for k in range(3):
            #         print(temp_offset[k][0,:,30,30])
            return agg_cls_out, pts_out_init[:1,...], pts_out_refine[:1,...]
        

    def forward(self, feats,test=False):
        #5 feature map
        self.reppoints=[]
        self.offset=[]
        self.inv_offset=[]
        outs=multi_apply(self.forward_single, feats,[0,1,2,3,4],[test for i in range(5)])

        # outs=[]
        # for i in range(len(feats)):
        #     outs.append(self.forward_single(feats[i],i))
        # outs=tuple(outs)
        return outs

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate.
        """
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine, stride,
                    num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine)

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)
        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # print(len(featmap_sizes),len(self.point_generators))
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if cfg.init.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = point_target(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg.init,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(
                    pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat(
                    [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center +
                            bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = point_target(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg.refine,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        # print('losses_cls:',torch.stack(losses_cls).mean().item())
        # print('loss_pts_init:',torch.stack(losses_pts_init).mean().item())
        # print('loss_pts_refine:',torch.stack(losses_pts_refine).mean().item())
        self.losses_cls=torch.stack(losses_cls).mean().item()
        self.loss_pts_init=torch.stack(losses_pts_init).mean().item()
        self.loss_pts_refine=torch.stack(losses_pts_refine).mean().item()
        return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True,index=False):
        assert len(cls_scores) == len(pts_preds_refine)
        bbox_preds_refine = [
            self.points2bbox(pts_pred_refine)
            for pts_pred_refine in pts_preds_refine
        ]
        num_levels = len(cls_scores)
        # print(num_levels)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if not index:
                proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale, nms,index)
                result_list.append(proposals)
            else:
                proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale, nms,index)
                result_list.append(proposals)
        if not index:
            return result_list
        else:
            return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True,index=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_index=[]
        det_index=[]
        
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, mlvl_points)):
            # torch.Size([2, 256, 48, 156])
            # torch.Size([2, 256, 24, 78])
            # torch.Size([2, 256, 12, 39])
            # torch.Size([2, 256, 6, 20])
            # torch.Size([2, 256, 3, 10])
            # if cls_score.size()[-1]!=78:
            #     continue
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # print(cls_score.shape)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:#True
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            #1000
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            loc=torch.ones(points.shape[0],3).to(cls_scores[0].device)
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            loc[:,:2]=points[:, :2]
            # point_strides=[8, 16, 32, 64, 128],
            loc[:,2]=self.point_strides[i_lvl]*loc[:,2]
            # loc[:,3]=scores*loc[:,3:]
            mlvl_index.append(loc)
        mlvl_index=torch.cat(mlvl_index)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            if not index:
                return det_bboxes, det_labels
            else:
                # print(det_bboxes.shape,mlvl_bboxes.shape)
                # torch.Size([64, 5]) torch.Size([1000, 4])
                # print(mlvl_index.shape)
                for i_nmsb in range(det_bboxes.shape[0]):
                    box=det_bboxes[i_nmsb,:4]
                    select=torch.where(mlvl_bboxes==box)
                    # print(select)
                    # (tensor([9, 9, 9, 9], device='cuda:0'), tensor([0, 1, 2, 3], device='cuda:0'))
                    # print(select[0][0],mlvl_index[select[0][0]])
                    # print(torch.max(mlvl_scores[select[0][0]]))
                    # print(mlvl_index[select[0][0]].view(-1))
                    det_index.append(torch.cat([mlvl_index[select[0][0]].view(-1),torch.max(mlvl_scores[select[0][0]]).view(-1)]))
                # print(det_index)
                # exit()
                if len(det_index)==0:
                    det_index.append(mlvl_index[0])
                # exit()
                return det_bboxes, det_labels,torch.stack(det_index,dim=0)
        else:
            return mlvl_bboxes, mlvl_scores
