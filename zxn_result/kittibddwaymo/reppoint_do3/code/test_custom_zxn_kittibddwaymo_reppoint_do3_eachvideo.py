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

def kitti_eval(det_results, dataset, iou_thr=0.5):
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for each in dataset:
        bboxes = np.array(each['ann']['bboxes'])
        labels = np.array(each['ann']['labels'])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
        # if level_selected_eval == 'object':
        #     occ_index = each['occ_index']
        #     pair_index = each['occ_index_pair']
        #     selected_index = occ_index + pair_index
        #     ignore_it = np.zeros(len(each['ann']['labels']), dtype=np.int32)
        #     for i in range(len(each['ann']['labels'])):
        #         if i not in selected_index:
        #             # print(each['ann']['labels'][i])
        #             ignore_it[i] = 1
        #         else:
        #             ignore_it[i] = 0
        #             # print(each['ann']['labels'][i])
        #     gt_ignore.append(ignore_it)

    if not gt_ignore:
        gt_ignore = None
    dataset_name = 'kitti'

    mean_ap, eval_results = eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)
    return mean_ap, eval_results


# annotation
# all / All : the whole datasets
# sub / Sub : the sub datasets (VV, PP, VP)
# level_selected: 'frame' or 'object'
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
jsonfile_name = 'kitti_bdd_waymo_2class_val.json'  #the whole datasets
occlusion_file_name = 'kitti_bdd_waymo_2class_val_vp.json'  # 'occlusion_test_v.json' or 'occlusion_test_p.json' or 'occlusion_test_vp.json'
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
out_path = '/home/zxn/RepPoints/zxn_result/debug/repoint_baseline_do3/kittibddwayo_epoch20_thres03_nms05_baseline_rerun_02142013_sub13'
if not os.path.exists(out_path):
    os.mkdir(out_path)
    os.mkdir(os.path.join(out_path, out_name))

compute_time = 0
support_count = 2

results = []
video_length = 0
video_name_check = None

scale = [8, 16, 32, 64, 128]
scale = {'8': 0, '16': 1, '32': 2, '64': 3, '128': 4}

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
            data_selected.append(frame)
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
            # data_selected.append(data_all_copy[data_all_filenames.index(img)])
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

        mmcv.dump(result_record, os.path.join(out_path, out_name, 'det_result_objlevel_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(data_selected, os.path.join(out_path, out_name, 'gt_selected_objlevel_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))

        mmcv.dump(loc_data, os.path.join(out_path, out_name, 'loc_result_objlevel_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(reppoint_data, os.path.join(out_path, out_name, 'reppoints_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))

        print('evaluating result of ', out_name)
        kitti_eval(result_record, data_selected)

elif test_datasets_version == 'Sub':

        detection_all = mmcv.load(os.path.join(out_path, out_name, 'det_result_val_all.pkl'))
        loc_all = mmcv.load(os.path.join(out_path, out_name, 'loc_result_val_all.pkl'))
        reppoints_all = mmcv.load(os.path.join(out_path, out_name, 'reppoints_val_all.pkl'))

        # 从所有的图像名称中筛选出次 video 的名称
        videos_list = []
        for each in data_all_filenames:
            if each[:-11] not in videos_list:
                videos_list.append(each[:-11])
            else:
                print('!!!!!!!!!!!!!')
        import xlsxwriter
        workbook = xlsxwriter.Workbook('Result_waymo_1510.xlsx')
        worksheet = workbook.add_worksheet()
        for video_index, each_video in enumerate(videos_list):
            reppoint_data = [[] for i in range(5)]
            if level_selected == 'object' or level_selected == 'objectandframe':
                level_selected_save = 'object'
                result_record_sub = []
                data_selected_sub = []
                loc_data_sub = []
                reppoint_data_sub = []
                for each_data in data_all:
                    if each_data['filename'][:-11] == each_video:
                        result_record_sub.append(detection_all[data_all_filenames.index(each_data['filename'])])
                        data_selected_sub.append(each_data)
                mean_ap, eval_results = kitti_eval(result_record_sub, data_selected_sub)

                class_names = []
                recall = []
                precision = []
                ap =[]
                num_gts = []
                num_dets = []
                for i, cls_result in enumerate(eval_results):
                    if i >= 2:
                        continue
                    # if cls_result['recall'].size > 0:
                    class_names.append(model.CLASSES[i])
                    recall.append(np.array(cls_result['recall'], ndmin=2)[:, -1])
                    precision.append(np.array(cls_result['precision'], ndmin=2)[:, -1])
                    ap.append(cls_result['ap'])
                    num_gts.append(cls_result['num_gts'])
                    num_dets.append(cls_result['num_dets'])

                print(num_gts, num_dets, recall, precision, ap)

                worksheet.write(video_index, 0, each_video)  # 记录video名字
                worksheet.write(video_index, 1, mean_ap)  # 记录video名字

                worksheet.write(video_index, 2, class_names[0])  # 记录 类别 名字
                worksheet.write(video_index, 3, num_gts[0])  # 记录 gts
                worksheet.write(video_index, 4, num_dets[0])  # 记录 dets
                worksheet.write(video_index, 5, recall[0])  # 记录 recall
                worksheet.write(video_index, 6, precision[0])  # 记录 precision
                worksheet.write(video_index, 7, ap[0])  # 记录 ap

                worksheet.write(video_index, 8, class_names[1])  # 记录 类别 名字
                worksheet.write(video_index, 9, num_gts[1])  # 记录 gts
                worksheet.write(video_index, 10, num_dets[1])  # 记录 dets
                worksheet.write(video_index, 11, recall[1])  # 记录 recall
                worksheet.write(video_index, 12, precision[1])  # 记录 precision
                worksheet.write(video_index, 13, ap[1])  # 记录 ap

        workbook.close()
        print('336 ')
        from IPython import embed
        embed()



else:
    print('Please select test_datasets_version: All or Sub ?')
    from IPython import embed
    embed()





