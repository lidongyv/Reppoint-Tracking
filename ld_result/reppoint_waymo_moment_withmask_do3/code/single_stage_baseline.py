import torch.nn as nn

from mmdet.core import bbox2result,loc2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from collections import OrderedDict
import torch
import numpy as np
@DETECTORS.register_module
class SingleStageDetector_Baseline(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 agg=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 index=False):
        super(SingleStageDetector_Baseline, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        self.index=True

    def init_weights(self, pretrained=None):
        super(SingleStageDetector_Baseline, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        # torch.Size([2, 3, 384, 1248])
        x = self.backbone(img)
        # torch.Size([2, 256, 96, 312])
        # torch.Size([2, 512, 48, 156])
        # torch.Size([2, 1024, 24, 78])
        # torch.Size([2, 2048, 12, 39])
        if self.with_neck:
            x = self.neck(x)
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        
        print(img.shape)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=None)

        return losses

    def simple_test(self, img, img_meta, rescale=False):

        print('single test')
        print(img.shape)
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        index=True
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs,index=index)
        # print(bbox_list[0][2])
        # print(bbox_list[0][:2])
        # # print(bbox_results)
        # exit()
        box_loc=bbox_list[0][2]
        bbox_list=[bbox_list[0][:2]]

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        loc_results = loc2result(box_loc, bbox_list[0][1], self.bbox_head.num_classes)
        return bbox_results[0],loc_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def simple_trackor(self, img, img_meta, rescale=False):
        print(img.shape)
        print('single eval')
        if img.shape[1]>3:
            n=img.shape[1]//3
            img=img.view(n,3,img.shape[2],img.shape[3])
            # print(((img[0]==img[1]).sum().float()/3)/(img.shape[-1]*img.shape[-2]))
            #0.1864
        # print(img.shape)
        
        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])
        x = self.extract_feat(img)
        

        outs = self.bbox_head(x,test=True)
        # print(len(outs))
        # print(len(outs[0]))
        # print(outs[0][0].shape)
        # exit()
        index=True
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs,index=index)
        # print(bbox_list[0][2])
        # print(bbox_list[0][:2])
        # # print(bbox_results)
        # exit()
        box_loc=bbox_list[0][2]
        bbox_list=[bbox_list[0][:2]]

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        loc_results = loc2result(box_loc, bbox_list[0][1], self.bbox_head.num_classes)
        return bbox_results[0],loc_results