import torch.nn as nn

from mmdet.core import bbox2result, loc2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from collections import OrderedDict
import torch
import numpy as np

from ..flownet import models, datasets
from ..flownet import losses as Loss
from ..flownet.utils import tools
from ..flownet.utils.flow_utils import make_color_wheel, compute_color, flow_to_image
import os

checkpoint = torch.load('./mmdetection/mmdet/models/flownet/FlowNet2_checkpoint.pth.tar')

import cv2


def set_id_grid(b, h, w):
    # b, _, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).float()  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).float()  # [1, H, W]
    ones = torch.ones(1, h, w).float()
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1).repeat(b, 1, 1, 1)  # [1, 3, H, W]
    return pixel_coords


# class ModelAndLoss(nn.Module):
#     def __init__(self):
#         super(ModelAndLoss, self).__init__()
#         self.model = model_class()
#         self.loss = loss_class()
#
#     def forward(self, data):
#         output = self.model(data)
#         return output

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

        self.index = True

        self.flownet_model = tools.module_to_dict(models)['FlowNet2']()
        self.flownet_model.load_state_dict(checkpoint['state_dict'])

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
        # # --- modified ---
        print('----------------------------------')

        [print(each['filename']) for each in img_metas]

        # 先从 8个里面弄出来5个把程序调试通，后面在考虑怎么弄数据
        now_img = img[1:2, :, :, :]
        former_img = img[:1, :, :, :]
        later_img = img[2:3, :, :, :]
        print(now_img.shape)
        print(former_img.shape)
        print(later_img.shape)

        # later_img = img[:, :, 256 * 2:256 * 3, :256]
        batch_size, c, im_h, im_w = now_img.shape

        base_feat = self.extract_feat(now_img)
        base_former_feat = self.extract_feat(former_img)
        base_later_feat = self.extract_feat(later_img)

        # 设置
        grid_pixel = set_id_grid(batch_size, int(im_h / 4), int(im_w / 4)).cuda()
        u_grid = grid_pixel[:, 0, :, :].view(batch_size, -1)
        u_norm = 2 * (u_grid / (int(im_w / 4) - 1)) - 1
        v_grid = grid_pixel[:, 1, :, :].view(batch_size, -1)
        v_norm = 2 * (v_grid / (int(im_h / 4) - 1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=2)
        pixel_coords_base_4 = pixel_coords.view(batch_size, int(im_h / 4), int(im_w / 4), 2)

        grid_pixel = set_id_grid(batch_size, int(im_h / 8), int(im_w / 8)).cuda()
        u_grid = grid_pixel[:, 0, :, :].view(batch_size, -1)
        u_norm = 2 * (u_grid / (int(im_w / 8) - 1)) - 1
        v_grid = grid_pixel[:, 1, :, :].view(batch_size, -1)
        v_norm = 2 * (v_grid / (int(im_h / 8) - 1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=2)
        pixel_coords_base_8 = pixel_coords.view(batch_size, int(im_h / 8), int(im_w / 8), 2)

        grid_pixel = set_id_grid(batch_size, int(im_h / 16), int(im_w / 16)).cuda()
        u_grid = grid_pixel[:, 0, :, :].view(batch_size, -1)
        u_norm = 2 * (u_grid / (int(im_w / 16) - 1)) - 1
        v_grid = grid_pixel[:, 1, :, :].view(batch_size, -1)
        v_norm = 2 * (v_grid / (int(im_h / 16) - 1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        pixel_coords_base_16 = pixel_coords.view(batch_size, int(im_h / 16), int(im_w / 16), 2)

        grid_pixel = set_id_grid(batch_size, int(im_h / 32), int(im_w / 32)).cuda()
        u_grid = grid_pixel[:, 0, :, :].view(batch_size, -1)
        u_norm = 2 * (u_grid / (int(im_w / 32) - 1)) - 1
        v_grid = grid_pixel[:, 1, :, :].view(batch_size, -1)
        v_norm = 2 * (v_grid / (int(im_h / 32) - 1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        pixel_coords_base_32 = pixel_coords.view(batch_size, int(im_h / 32), int(im_w / 32), 2)

        grid_pixel = set_id_grid(batch_size, int(im_h / 64), int(im_w / 64)).cuda()
        u_grid = grid_pixel[:, 0, :, :].view(batch_size, -1)
        u_norm = 2 * (u_grid / (int(im_w / 64) - 1)) - 1
        v_grid = grid_pixel[:, 1, :, :].view(batch_size, -1)
        v_norm = 2 * (v_grid / (int(im_h / 64) - 1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        pixel_coords_base_64 = pixel_coords.view(batch_size, int(im_h / 64), int(im_w / 64), 2)

        grid_pixel = set_id_grid(batch_size, int(im_h / 128), int(im_w / 128)).cuda()
        u_grid = grid_pixel[:, 0, :, :].view(batch_size, -1)
        u_norm = 2 * (u_grid / (int(im_w / 128) - 1)) - 1
        v_grid = grid_pixel[:, 1, :, :].view(batch_size, -1)
        v_norm = 2 * (v_grid / (int(im_h / 128) - 1)) - 1
        pixel_coords = torch.stack([u_norm, v_norm], dim=1)
        pixel_coords_base_128 = pixel_coords.view(batch_size, int(im_h / 128), int(im_w / 128), 2)

        std_vector = torch.from_numpy(np.array([58.395, 57.12, 57.375])).unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1).cuda().float()
        mean_vector = torch.from_numpy(np.array([123.675, 116.28, 103.53])).unsqueeze(0).unsqueeze(-1).unsqueeze(
            -1).cuda().float()

        now_img = now_img * std_vector + mean_vector
        former_img = former_img * std_vector + mean_vector
        later_img = later_img * std_vector + mean_vector

        # now_img = (now_img - now_img.min()) / (now_img.max() - now_img.min()) * 255
        # former_img = (former_img - former_img.min()) / (former_img.max() - former_img.min()) * 255
        # later_img = (later_img - later_img.min()) / (later_img.max() - later_img.min()) * 255

        former_sample_pair = torch.cat([former_img.unsqueeze(2), now_img.unsqueeze(2)], 2)
        later_sample_pair = torch.cat([now_img.unsqueeze(2), later_img.unsqueeze(2)], 2)
        former_flow = self.flownet_model(former_sample_pair).detach()
        later_flow = self.flownet_model(later_sample_pair).detach()

        now_a = now_img[0].permute(1, 2, 0)
        former_a = former_img[0].permute(1, 2, 0)
        later_a = later_img[0].permute(1, 2, 0)

        now_b = now_a.detach().cpu().numpy()[:, :, ::-1]
        former_b = former_a.detach().cpu().numpy()[:, :, ::-1]
        later_b = later_a.detach().cpu().numpy()[:, :, ::-1]

        former_name = img_metas[0]['filename'].split('/')[-1].split('.')[0]
        now_name = img_metas[1]['filename'].split('/')[-1].split('.')[0]
        later_name = img_metas[2]['filename'].split('/')[-1].split('.')[0]
        if later_name == '000197':
            from IPython import embed
            embed()


        if not os.path.exists('./temp/%s'%former_name):
            os.makedirs('./temp/%s' % former_name)
        if not os.path.exists('./temp/%s'%now_name):
            os.makedirs('./temp/%s' % now_name)

        cv2.imwrite('./temp/img_now_%s.png' % now_name, now_b)
        cv2.imwrite('./temp/img_former_%s.png' % former_name, former_b)
        cv2.imwrite('./temp/img_later_%s.png' % later_name, later_b)

        np.save('./temp/%s/flow_%sto_%s.npy' % (former_name, former_name, now_name), former_flow.detach().cpu().numpy())
        np.save('./temp/%s/flow_%sto_%s.npy' % (now_name, now_name, later_name), later_flow.detach().cpu().numpy())

        #
        for each_index, each_flow in enumerate(former_flow):
            a = each_flow.permute(1, 2, 0)
            flow_image_former_np = flow_to_image(a)
            cv2.imwrite('./temp/%s/flow_%sto_%s.png' % (former_name, former_name, now_name), flow_image_former_np)
        # 保存原始大小的光流  --- later

        for each_index, each_flow in enumerate(later_flow):
            b = each_flow.permute(1, 2, 0)
            flow_image_later_np = flow_to_image(b)
            cv2.imwrite('./temp/%s/flow_%sto_%s.png' % (now_name, now_name, later_name), flow_image_later_np)


        with torch.no_grad():
            x = self.extract_feat(img)
        # print(img.shape)
        outs = self.bbox_head(x)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=None)
        return losses

    def simple_test(self, img, img_meta, rescale=False):

        print('single test')

        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=True)
        index = True
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs, index=index)
        # print(bbox_list[0][2])
        # print(bbox_list[0][:2])
        # # print(bbox_results)
        # exit()
        box_loc = bbox_list[0][2]
        bbox_list = [bbox_list[0][:2]]

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        loc_results = loc2result(box_loc, bbox_list[0][1], self.bbox_head.num_classes)
        return bbox_results[0], loc_results

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def simple_trackor(self, img, img_meta, rescale=False):
        print(img.shape)
        print('single eval')

        if img.shape[1] > 3:
            n = img.shape[1] // 3
            img = img.view(n, 3, img.shape[2], img.shape[3])
            # print(((img[0]==img[1]).sum().float()/3)/(img.shape[-1]*img.shape[-2]))
            # 0.1864
        # print(img.shape)

        # torch.Size([2, 256, 48, 156])
        # torch.Size([2, 256, 24, 78])
        # torch.Size([2, 256, 12, 39])
        # torch.Size([2, 256, 6, 20])
        # torch.Size([2, 256, 3, 10])

        print('290 debug')
        from IPython import embed
        embed()

        x = self.extract_feat(img)

        outs = self.bbox_head(x, test=True)
        # print(len(outs))
        # print(len(outs[0]))
        # print(outs[0][0].shape)
        # exit()
        index = True
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs, index=index)
        # print(bbox_list[0][2])
        # print(bbox_list[0][:2])
        # # print(bbox_results)
        # exit()
        box_loc = bbox_list[0][2]
        bbox_list = [bbox_list[0][:2]]

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        loc_results = loc2result(box_loc, bbox_list[0][1], self.bbox_head.num_classes)
        return bbox_results[0], loc_results