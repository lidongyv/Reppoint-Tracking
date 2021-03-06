import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from mmdet.core import aggmulti_apply
from collections import OrderedDict
import torch
import numpy as np
@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
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
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.agg_check=agg
        if agg is not None:
            self.agg=builder.build_agg(agg)
        self.index=index
    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
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

        if self.agg_check:
            # x,trans_loss=self.agg(x)
            x=self.agg(x)
        if isinstance(x, tuple):
            outs = self.bbox_head(x)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
            losses = self.bbox_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # return losses,trans_loss
            return losses
        else:
            losses_all=[]
            # print('list')
            #[tuple(agg_output),tuple(refer_out),tuple(support1_out),tuple(support1_out)]
            for i in range(len(x)):
                outs = self.bbox_head(x[i])
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
                losses = self.bbox_head.loss(
                    *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
                losses_all.append(losses)
                # continue
            return losses_all


    def simple_test(self, img, img_meta, rescale=False):
        print(img.shape)
        print('single test')
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
        if self.agg_check:
            x=self.agg.forward_test(x)
        # agg_load=np.load('/home/ld/RepPoints/offset/agg_st_support/2/agg_f.npy')
        # agg=torch.from_numpy(agg_load).to(img.device)
        # print('agg check in single stage',(x[0]==agg).all())
        # load=[]
        # for i in range(len(x)):
        #     # print(x[i].shape)
        #     if i==0:
        #         load.append(agg)
        #     else:
        #         load.append(x[i])
        # x=tuple(load)
        
        outs = self.bbox_head(x)
        index=self.index
        index=True
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs,index=index)
        if index:
            box_loc=bbox_list[0][2]
            bbox_list=[bbox_list[0][:2]]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        if index:
            return bbox_results[0],box_loc
        else:
            return bbox_results[0]

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
        if self.agg_check:
            x=self.agg.forward_eval(x)
        if isinstance(x, tuple):
            
            outs = self.bbox_head(x)
            index=self.index
            index=True
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs,index=index)
            if index:
                box_loc=bbox_list[0][2]
                bbox_list=[bbox_list[0][:2]]
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                for det_bboxes, det_labels in bbox_list
            ]
            if index:
                return bbox_results[0],box_loc
            else:
                return bbox_results[0]
        else:
            out=[]
            # length 12: out=[tuple(refer_out),tuple(agg_out)]+support_out
            for i in range(len(x)):
                outs = self.bbox_head(x[i])
                index=self.index
                index=True
                bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
                bbox_list = self.bbox_head.get_bboxes(*bbox_inputs,index=index)
                if index:
                    box_loc=bbox_list[0][2]
                    bbox_list=[bbox_list[0][:2]]
                bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                    for det_bboxes, det_labels in bbox_list
                ]
                if index:
                    out.append([bbox_results[0],box_loc])
                else:
                    out.append(bbox_results[0])
            print(len(out))
            return out