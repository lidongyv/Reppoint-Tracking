# -*- coding: utf-8 -*- 
# @Author: Lidong Yu
# @Date: 2019-11-25 19:24:06
# @Last Modified by: Lidong Yu
# @Last Modified time: 2019-11-25 19:24:06

from mmdet.apis import init_detector, inference_detector, show_result, inference_trackor
import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import cv2
import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core import eval_map
import os


# def kitti_eval(det_results, dataset, iou_thr=0.5):
#     gt_bboxes = []
#     gt_labels = []
#     gt_ignore = []
#     for i in range(len(dataset)):
#         ann = dataset.get_ann_info(i)
#         bboxes = ann['bboxes']
#         labels = ann['labels']
#         gt_bboxes.append(bboxes)
#         gt_labels.append(labels)
#     # if i>10:
#     # 	break
#     if not gt_ignore:
#         gt_ignore = None
#
#     dataset_name = 'kitti'
#     print('51 debug')
#     from IPython import embed
#     embed()
#     eval_map(
#         det_results,
#         gt_bboxes,
#         gt_labels,
#         gt_ignore=gt_ignore,
#         scale_ranges=None,
#         iou_thr=iou_thr,
#         dataset=dataset_name,
#         print_summary=True)

def kitti_eval(det_results, dataset, iou_thr=0.5):
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for each in dataset:
        bboxes = np.array(each['ann']['bboxes'])
        labels = np.array(each['ann']['labels'])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = None
    dataset_name = 'kitti'
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


config_file = '/home/zxn/RepPoints/zxn_result/kittibddwaymo/code/repoint_baseline_do3.py'
checkpoint_file = '/home/zxn/RepPoints/zxn_result/kittibddwaymo/models/epoch_20.pth'

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
data_path = '/backdata01/'
jsonfile_name_all = 'kitti_bdd_waymo_2class_val.json'
occlusion_file_name = 'kitti_bdd_waymo_2class_val.json'
# test a video and show the results

with open(data_path + jsonfile_name_all, 'r') as f:
    data_all = json.load(f)

with open(data_path + occlusion_file_name, 'r') as f:
    datas = json.load(f)
if len(datas) > 10:
    datas = [datas]
#将所有frame的名称都记录到一个list里面
data_all_filenames = [each['filename'] for each in data_all]

# occlusion_file_name = 'occlusion_test_v.json'

# with open('/backdata01/KITTI/kitti/tracking/' + occlusion_file_name, 'r') as f:
#     datas = json.load(f)

# with open(data_path + '/' + 'kitti_bdd_waymo_2class_val_v.json', 'r') as f:
#     datas = json.load(f)
# with open(os.path.join(data_path, jsonfile_name), 'r') as f:
#     data = json.load(f)
compute_time = 0
support_count = 2
out_name = 'baseline'
out_path = '/home/zxn/RepPoints/zxn_result/debug/repoint_baseline_do3/kittibddwayo_epoch20_thres01_nms05_baseline'
if not os.path.exists(out_path):
    os.mkdir(out_path)
    os.mkdir(os.path.join(out_path, out_name))
results = []
video_length = 0
video_name_check = None
result_record = []

eval_data = []

loc_data = []

scale = [8, 16, 32, 64, 128]
scale = {'8': 0, '16': 1, '32': 2, '64': 3, '128': 4}

offsets = [[] for i in range(5)]
# offsets=[]
# load and test

# result_record=mmcv.load(os.path.join(out_path,'det_result.pkl'))
# print('evaluating result of support', )
# print(result_record)
# kitti_eval(result_record, dataset)
# exit()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.CLASSES = dataset.CLASSES
# result_record=[]
if len(datas) == 1:
    ratio_array = ['all']
if len(datas) == 10:
    ratio_array = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
for ratio_index, data in enumerate(datas):
    data_selected = []
    reppoint_data = [[] for i in range(5)]
    for i, (frame) in enumerate(data):
        print(i, 'in', len(data))
        video_name = frame['video_id']
        if video_name_check is None:
            video_name_check = video_name
        else:
            if video_name_check == video_name:
                video_length += 1
            else:
                video_name_check = video_name
                video_length = 0
        print('video_name', video_name, 'image_name', frame['filename'], 'video_length', video_length)
        img_name = frame['filename']

        # img = mmcv.imread(os.path.join(data_path,img_name))

        img = os.path.join(data_path, img_name)
        data_selected.append(frame)
        # data_selected.append(data_all[data_all_filenames.index(img)])
        img_list = img
        result = inference_trackor(model, img_list)
        # for m in range(5):
        #     offsets[m].append(model.bbox_head.offset[m])
        bbox_result = result[0]

        loc_result = result[1]

        result_record.append(bbox_result)
        loc_data.append(loc_result)
        loc_result = loc_result.long()
        # four value and one score
        loc_result = loc_result.cpu()
        bboxes = np.vstack(bbox_result)
        loc = np.vstack(loc_result)
        print(bboxes.shape)
        print(loc.shape)
        scores = bboxes[:, -1]
        inds = scores > 0
        scores = bboxes[inds, :][:, 4:]
        bboxes = bboxes[inds, :][:, :4]

        reppoint_t = model.bbox_head.reppoints
        for m in range(len(reppoint_t)):
            reppoint_data[m].append(reppoint_t)
        # from IPython import embed; embed()
        # print("offset_length", len(model.bbox_head.offset))
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        labels = labels[inds]
        frame_data = {"video_id": frame['video_id'], "filename": os.path.join(frame['filename']), \
                      "ann": {"bboxes": bboxes.tolist(), "labels": labels.tolist(), \
                              "track_id": labels.tolist(), 'score': scores.tolist()}}
        eval_data.append(frame_data)


    mmcv.dump(result_record, os.path.join(out_path, out_name, 'det_result_%s_%s.pkl'%(
        occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
    mmcv.dump(loc_data, os.path.join(out_path, out_name, 'loc_result_%s_%s.pkl'%(
        occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
    mmcv.dump(reppoint_data, os.path.join(out_path, out_name, 'reppoints_%s_%s.pkl'%(
        occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
    print('evaluating result of ', out_name)
    kitti_eval(result_record, data_selected)

    print('-----------------------------')
    from IPython import embed
    embed()

