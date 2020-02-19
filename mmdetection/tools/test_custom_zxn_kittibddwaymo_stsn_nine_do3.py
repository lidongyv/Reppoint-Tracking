# -*- coding: utf-8 -*-
# @Author: Lidong Yu
# @Date: 2019-11-25 19:24:06
# @Last Modified by: Lidong Yu
# @Last Modified time: 2019-11-25 19:24:06
import copy
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
from tqdm import tqdm

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
def kitti_eval_base(det_results, dataset, iou_thr=0.5):
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

def kitti_eval(det_results, dataset, level_selected_eval, iou_thr=0.5):
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for each in dataset:
        bboxes = np.array(each['ann']['bboxes'])
        labels = np.array(each['ann']['labels'])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
        if level_selected_eval == 'object':
            occ_index = each['occ_index']
            pair_index = each['occ_index_pair']
            selected_index = occ_index + pair_index
            ignore_it = np.zeros(len(each['ann']['labels']), dtype=np.int32)
            for i in range(len(each['ann']['labels'])):
                if i not in selected_index:
                    # print(each['ann']['labels'][i])
                    ignore_it[i] = 1
                else:
                    ignore_it[i] = 0
                    # print(each['ann']['labels'][i])
            gt_ignore.append(ignore_it)

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


# annotation
# all / All : the whole datasets
# sub / Sub : the sub datasets (VV, PP, VP)
# level_selected: 'frame' or 'object'
config_file = '/home/zxn/RepPoints/zxn_result/kittibddwaymo/stsn_nine_do3/code/stsn_do3.py'
checkpoint_file = '/home/zxn/RepPoints/zxn_result/kittibddwaymo/stsn_nine_do3/epoch_30.pth'

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True

dataset = build_dataset(cfg.data.test)
data_path = '/backdata01/'
jsonfile_name = 'kitti_bdd_waymo_2class_val_13.json'  #the whole datasets
occlusion_file_name = 'kitti_bdd_waymo_2class_val_13.json'  # 'occlusion_test_v.json' or 'occlusion_test_p.json' or 'occlusion_test_vp.json'
level_selected = 'object'  # 'frame' or 'object' or 'objectandframe'

with open(data_path + jsonfile_name, 'r') as f:
    data_all = json.load(f)
data_all_filenames = [each['filename'] for each in data_all]

with open(data_path + occlusion_file_name, 'r') as f:
    datas = json.load(f)

# len(datas) != 10 就意味着是 all 的数据
if len(datas) > 10:
    datas = [datas]
if len(datas) == 1:
    ratio_array = ['all']
    test_datasets_version = 'All'
if len(datas) == 10:
    ratio_array = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9', 'sub_all']
    sub_all_detected = []  # for recording detected
    sub_all_selected = []  # for recording gt
    test_datasets_version = 'Sub'

    if level_selected == 'object' or level_selected == 'objectandframe':
        datas_object = datas.copy()
        datas_object[-1] = datas_object[0] + datas_object[1] + datas_object[2] + datas_object[3] + datas_object[4] + datas_object[5] + datas_object[6] + datas_object[7] + datas_object[8]
    if level_selected == 'frame' or level_selected == 'objectandframe':
        datas_frame = datas.copy()
        temp_list = []
        for c_i in range(9):
            for c_j in range(len(datas_frame[c_i])):
                if datas_frame[c_i][c_j]['filename'] not in temp_list:
                    temp_list.append(datas_frame[c_i][c_j]['filename'])
                    datas_frame[-1].append(datas_frame[c_i][c_j])

out_name = 'baseline'
out_path = '/home/zxn/RepPoints/zxn_result/debug/stsn_nine_do3/kittibddwayo_epoch30_thres03_nms05_baseline'

if not os.path.exists(out_path):
    os.mkdir(out_path)
    os.mkdir(os.path.join(out_path, out_name))

results=[]
video_length=0
video_name_check=None
result_record=[]

eval_data=[]

loc_data=[]
offset_data=[[] for i in range(5)]
offset_data=[copy.deepcopy(offset_data),copy.deepcopy(offset_data)]
# offset_data=[offset_data.copy(),offset_data.copy()]

img_record=[]
scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}


compute_time = 0
support_count = 2


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.CLASSES = dataset.CLASSES
# result_record=[]

if test_datasets_version == 'All':

    result_record = []
    data_selected = []
    eval_data = []
    loc_data = []
    reppoint_data = [[] for i in range(5)]
    for ratio_index, data in tqdm(enumerate(datas)):
        data_all_copy = data_all.copy()
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
            img_list = [img]

            if data[i - 2]['video_id'] == video_name:
                img_list.append(os.path.join(data_path, data[i - 2]['filename']))
            else:
                img_list.append(os.path.join(data_path, data[i]['filename']))
                print('start')
            if i + 2 >= len(data):
                img_list.append(os.path.join(data_path, data[i]['filename']))
            else:
                if data[i + 2]['video_id'] == video_name:
                    img_list.append(os.path.join(data_path, data[i + 2]['filename']))
                else:
                    img_list.append(os.path.join(data_path, data[i]['filename']))
                    print('end')
            img_record.append(img_list)

            result = inference_trackor(model, img_list)
            offset_t = model.bbox_head.offset

            bbox_result = result[0]
            loc_result = result[1]
            result_record.append(bbox_result)
            loc_data.append(loc_result)
            for m in range(len(offset_t)):
                for n in range(len(offset_t[m])):
                    offset_data[n][m].append(offset_t[m][n])
        print('--------------------------- 207 before save ---------------------------')
        from IPython import embed
        embed()
        mmcv.dump(img_record, os.path.join(out_path, out_name, 'images_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(offset_data, os.path.join(out_path, out_name, 'offset_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(result_record, os.path.join(out_path, out_name, 'det_result_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(loc_data, os.path.join(out_path, out_name, 'loc_result_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))

        print('evaluating result of ', out_name)
        kitti_eval_base(result_record, data_selected)

elif test_datasets_version == 'Sub':

        detection_all = mmcv.load(os.path.join(out_path, out_name, 'det_result_val_all.pkl'))
        loc_all = mmcv.load(os.path.join(out_path, out_name, 'loc_result_val_all.pkl'))
        reppoints_all = mmcv.load(os.path.join(out_path, out_name, 'reppoints_val_all.pkl'))

        eval_data = []
        loc_data = []
        reppoint_data = [[] for i in range(5)]
        if level_selected == 'object' or level_selected == 'objectandframe':
            level_selected_eval = 'object'
            for ratio_index, data in tqdm(enumerate(datas_object)):
                result_record_sub = []
                data_selected_sub = []
                loc_data_sub = []
                reppoint_data_sub = []
                data_copy = data.copy()
                for i, (frame) in enumerate(data_copy):
                    result_record_sub.append(detection_all[data_all_filenames.index(frame['filename'])])
                    data_selected_sub.append(frame)
                mmcv.dump(result_record_sub, os.path.join(out_path, out_name, 'det_result_objlevel_%s_%s_%s.pkl' % (
                    occlusion_file_name.split('.')[0].split('_')[-1], level_selected_eval, ratio_array[ratio_index])))
                mmcv.dump(data_selected_sub, os.path.join(out_path, out_name, 'gt_selected_objlevel_%s_%s_%s.pkl' % (
                    occlusion_file_name.split('.')[0].split('_')[-1], level_selected_eval, ratio_array[ratio_index])))
                # mmcv.dump(loc_data, os.path.join(out_path, out_name, 'loc_result_objlevel_%s_%s.pkl' % (
                #     occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
                # mmcv.dump(reppoint_data, os.path.join(out_path, out_name, 'reppoints_%s_%s.pkl' % (
                #     occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
                kitti_eval(result_record_sub, data_selected_sub, level_selected_eval)
        print('------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------')
        if level_selected == 'frame' or level_selected == 'objectandframe':
            level_selected_eval = 'frame'
            for ratio_index, data in tqdm(enumerate(datas_frame)):
                result_record_sub = []
                data_selected_sub = []
                loc_data_sub = []
                reppoint_data_sub = []
                data_copy = data.copy()
                for i, (frame) in enumerate(data_copy):
                    result_record_sub.append(detection_all[data_all_filenames.index(frame['filename'])])
                    data_selected_sub.append(frame)
                mmcv.dump(result_record_sub, os.path.join(out_path, out_name, 'det_result_objlevel_%s_%s_%s.pkl' % (
                    occlusion_file_name.split('.')[0].split('_')[-1], level_selected_eval, ratio_array[ratio_index])))
                mmcv.dump(data_selected_sub, os.path.join(out_path, out_name, 'gt_selected_objlevel_%s_%s_%s.pkl' % (
                    occlusion_file_name.split('.')[0].split('_')[-1], level_selected_eval, ratio_array[ratio_index])))
                # mmcv.dump(loc_data, os.path.join(out_path, out_name, 'loc_result_objlevel_%s_%s.pkl' % (
                #     occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
                # mmcv.dump(reppoint_data, os.path.join(out_path, out_name, 'reppoints_%s_%s.pkl' % (
                #     occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
                kitti_eval(result_record_sub, data_selected_sub, level_selected_eval)


else:
    print('Please select test_datasets_version: All or Sub ?')
    from IPython import embed
    embed()





