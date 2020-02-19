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
from mmcv.image import imread, imwrite
from mmcv.utils import is_str
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

from enum import Enum

# annotation
# all / All : the whole datasets
# sub / Sub : the sub datasets (VV, PP, VP)
# level_selected: 'frame' or 'object'
config_file = '/home/zxn/RepPoints/zxn_result/kittibddwaymo/reppoint_do3/code/repoint_baseline_do3.py'
checkpoint_file = '/home/zxn/RepPoints/zxn_result/kittibddwaymo/reppoint_do3/epoch_20.pth'

class Color(Enum):
	"""An enum that defines common colors.

	Contains red, green, blue, cyan, yellow, magenta, white and black.
	"""
	red = (0, 0, 255)
	green = (0, 255, 0)
	blue = (255, 0, 0)
	cyan = (255, 255, 0)
	yellow = (0, 255, 255)
	magenta = (255, 0, 255)
	white = (255, 255, 255)
	black = (0, 0, 0)

def color_val(color):
	# """Convert various input to color tuples.
    #
	# Args:
	# 	color (:obj:`Color`/str/tuple/int/ndarray): Color inputs
    #
	# Returns:
	# 	tuple[int]: A tuple of 3 integers indicating BGR channels.
	# """

    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert channel >= 0 and channel <= 255
        return color
    elif isinstance(color, int):
        assert color >= 0 and color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError('Invalid type for color: {}'.format(type(color)))

def imshow_det_bboxes(img,
					  bboxes,
					  labels,
					  class_names=None,
					  score_thr=0,
					  bbox_color='green',
					  text_color='green',
					  thickness=1,
					  font_scale=0.5,
					  show=True,
					  win_name='',
					  wait_time=0,
					  out_file=None):
	"""Draw bboxes and class labels (with scores) on an image.

	Args:
		img (str or ndarray): The image to be displayed.
		bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
			(n, 5).
		labels (ndarray): Labels of bboxes.
		class_names (list[str]): Names of each classes.
		score_thr (float): Minimum score of bboxes to be shown.
		bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
		text_color (str or tuple or :obj:`Color`): Color of texts.
		thickness (int): Thickness of lines.
		font_scale (float): Font scales of texts.
		show (bool): Whether to show the image.
		win_name (str): The window name.
		wait_time (int): Value of waitKey param.
		out_file (str or None): The filename to write the image.
	"""
	assert bboxes.ndim == 2
	assert labels.ndim == 1
	assert bboxes.shape[0] == labels.shape[0]
	assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
	img = imread(img)

	bbox_color = color_val(bbox_color)
	text_color = color_val(text_color)

	for bbox, label in zip(bboxes, labels):
		bbox_int = bbox.astype(np.int32)
		left_top = (bbox_int[0], bbox_int[1])
		right_bottom = (bbox_int[2], bbox_int[3])
		cv2.rectangle(
			img, left_top, right_bottom, bbox_color, thickness=thickness)
		label_text = class_names[
			label] if class_names is not None else 'cls {}'.format(label)
		if len(bbox) > 4:
			label_text += '|{:.02f}'.format(bbox[-1])
		cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
					cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

	if show:
		imshow(img, win_name, wait_time)
	if out_file is not None:
		imwrite(img, out_file)


# cfg = mmcv.Config.fromfile(config_file)
# # set cudnn_benchmark
# if cfg.get('cudnn_benchmark', False):
#     torch.backends.cudnn.benchmark = True
# cfg.model.pretrained = None
# cfg.data.test.test_mode = True
# dataset = build_dataset(cfg.data.test)
#
# occlusion_file_name = 'kitti_bdd_waymo_2class_val_p.json'  # 'occlusion_test_v.json' or 'occlusion_test_p.json' or 'occlusion_test_vp.json'
# level_selected = 'frame'  # 'frame' or 'object' or 'objectandframe'
#
# with open(data_path + jsonfile_name, 'r') as f:
#     data_all = json.load(f)
# data_all_filenames = [each['filename'] for each in data_all]
#
# with open(data_path + occlusion_file_name, 'r') as f:
#     datas = json.load(f)

# # len(datas) != 10 就意味着是 all 的数据
# if len(datas) > 10:
#     datas = [datas]
# if len(datas) == 1:
#     ratio_array = ['all']
#     test_datasets_version = 'All'
# if len(datas) == 10:
#     ratio_array = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9', 'sub_all']
#     sub_all_detected = []  # for recording detected
#     sub_all_selected = []  # for recording gt
#     test_datasets_version = 'Sub'
#
#     if level_selected == 'object' or level_selected == 'objectandframe':
#         datas_object = datas.copy()
#         datas_object[-1] = datas_object[0] + datas_object[1] + datas_object[2] + datas_object[3] + datas_object[4] + datas_object[5] + datas_object[6] + datas_object[7] + datas_object[8]
#     if level_selected == 'frame' or level_selected == 'objectandframe':
#         datas_frame = datas.copy()
#         temp_list = []
#         for c_i in range(9):
#             for c_j in range(len(datas_frame[c_i])):
#                 if datas_frame[c_i][c_j]['filename'] not in temp_list:
#                     temp_list.append(datas_frame[c_i][c_j]['filename'])
#                     datas_frame[-1].append(datas_frame[c_i][c_j])
# loading detection results
out_name = 'det_result_13_all.pkl'
out_path = '/home/zxn/RepPoints/zxn_result/debug/repoint_baseline_do3/kittibddwayo_epoch20_thres03_nms05_baseline_rerun_02142013_sub13/baseline'
detection_all = mmcv.load(os.path.join(out_path, out_name))
# loading ground truth
data_path = '/backdata01/'
jsonfile_name = 'kitti_bdd_waymo_2class_val_13.json'  #the whole datasets

if not os.path.exists(os.path.join(out_path, 'all_detection_vis')):
    os.mkdir(os.path.join(out_path, 'all_detection_vis'))

color_list = ['green', 'red']
with open(data_path + jsonfile_name, 'r') as f:
    groundtruth_all = json.load(f)
class_names = ['Car', 'Person']
for index, each_gt in enumerate(groundtruth_all):
    each_img = each_gt['filename']
    mmcv.imread(each_gt['filename'])
    each_det_labels_list = []
    each_det = detection_all[index]
    for cls_index, cls_name in enumerate(class_names):

        each_det_bboxes = each_det[cls_index]
        each_det_labels = np.zeros(each_det_bboxes.shape[0]).astype(np.int) + cls_index
        each_det_labels_list.append(each_det_labels)
        # each_det_labels = each_det[]
        if 'kitti' in each_img.split('/'):
            out_file = os.path.join(out_path, 'all_detection_vis','kitti_'+ each_img.split('/')[-2], each_img.split('/')[-1])
        if 'bdd' in each_img.split('/'):
            out_file = os.path.join(out_path, 'all_detection_vis','bdd_'+ each_img.split('/')[-2], each_img.split('/')[-1])
        if 'waymo' in each_img.split('/'):
            out_file = os.path.join(out_path, 'all_detection_vis','waymo_'+ each_img.split('/')[-2], each_img.split('/')[-1])
        # for each_det_box in each_det_bboxes:
        #     cv2.rectangle(each_img, [], [], (0, 255, 0), 2)
        #     cv2.putText(image, text, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        #     cv2.putText(each_img, [], [], (0, 255, 0), 2)
    imshow_det_bboxes(
        each_img,
        np.vstack(each_det),
        np.hstack(each_det_labels_list),
        class_names=class_names,
        show=False,
        bbox_color='green',
        text_color='green',
        out_file=out_file)




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
            # loc_result = loc_result.long()
            # four value and one score
            # loc_result = loc_result.cpu()
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

        print('--------------------------- 207 before save ---------------------------')
        from IPython import embed
        embed()

        mmcv.dump(result_record, os.path.join(out_path, out_name, 'det_result_objlevel_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(data_selected, os.path.join(out_path, out_name, 'gt_selected_objlevel_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))

        mmcv.dump(loc_data, os.path.join(out_path, out_name, 'loc_result_objlevel_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))
        mmcv.dump(reppoint_data, os.path.join(out_path, out_name, 'reppoints_%s_%s.pkl'%(
            occlusion_file_name.split('.')[0].split('_')[-1], ratio_array[ratio_index])))

        print('evaluating result of ', out_name)
        kitti_eval_base(result_record, data_selected)






