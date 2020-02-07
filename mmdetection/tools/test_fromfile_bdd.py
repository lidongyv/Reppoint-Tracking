# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-25 19:24:06  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-25 19:24:06

from mmdet.apis import init_detector, inference_detector, show_result,inference_trackor
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
from mmcv.image import imread, imwrite
from mmcv.utils import is_str
from enum import Enum
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
def imshow(img, win_name='', wait_time=0):
	"""Show an image.

	Args:
		img (str or ndarray): The image to be displayed.
		win_name (str): The window name.
		wait_time (int): Value of waitKey param.
	"""
	cv2.imshow(win_name, imread(img))
	cv2.waitKey(wait_time)
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
	"""Convert various input to color tuples.

	Args:
		color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

	Returns:
		tuple[int]: A tuple of 3 integers indicating BGR channels.
	"""
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
					  out_file=None,out=False):
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
	if out:
		return img
def show_result(img,
				result,
				class_names,
				score_thr=0.1,
				wait_time=0,
				show=True,
				out_file=None):
	"""Visualize the detection results on the image.

	Args:
		img (str or np.ndarray): Image filename or loaded image.
		result (tuple[list] or list): The detection result, can be either
			(bbox, segm) or just bbox.
		class_names (list[str] or tuple[str]): A list of class names.
		score_thr (float): The threshold to visualize the bboxes and masks.
		wait_time (int): Value of waitKey param.
		show (bool, optional): Whether to show the image with opencv or not.
		out_file (str, optional): If specified, the visualization result will
			be written to the out file instead of shown in a window.

	Returns:
		np.ndarray or None: If neither `show` nor `out_file` is specified, the
			visualized image is returned, otherwise None is returned.
	"""
	assert isinstance(class_names, (tuple, list))
	img = mmcv.imread(img)
	img = img.copy()

	bbox_result = result
	bboxes = np.vstack(bbox_result)
	# draw bounding boxes
	labels = [
		np.full(bbox.shape[0], i, dtype=np.int32)
		for i, bbox in enumerate(bbox_result)
	]
	labels = np.concatenate(labels)
	imshow_det_bboxes(
		img,
		bboxes,
		labels,
		class_names=class_names,
		show=show,
		wait_time=wait_time,
		out_file=out_file)
	if not (show or out_file):
		return img
def bbox_overlaps(bboxes1, bboxes2):
	bboxes1 = bboxes1.astype(np.float32)
	bboxes2 = bboxes2.astype(np.float32)
	if bboxes1.shape[1]>4:
		bboxes1=bboxes1[:,:4]
		bboxes2=bboxes2[:,:4]
	rows = bboxes1.shape[0]
	cols = bboxes2.shape[0]
	ious = np.zeros((rows, cols), dtype=np.float32)
	if rows * cols == 0:
		return ious
	exchange = False
	if bboxes1.shape[0] > bboxes2.shape[0]:
		bboxes1, bboxes2 = bboxes2, bboxes1
		ious = np.zeros((cols, rows), dtype=np.float32)
		exchange = True
	area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
		bboxes1[:, 3] - bboxes1[:, 1] + 1)
	area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
		bboxes2[:, 3] - bboxes2[:, 1] + 1)
	for i in range(bboxes1.shape[0]):
		x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
		y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
		x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
		y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
		overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
			y_end - y_start + 1, 0)
		union = area1[i] + area2 - overlap
		ious[i, :] = overlap / union
	if exchange:
		ious = ious.T
	return ious
import os
def get_cls_results(det_results, gt_bboxes, gt_labels, gt_ignore, class_id):
	"""Get det results and gt information of a certain class."""
	cls_dets = [det[class_id]
				for det in det_results]  # det bboxes of this class
	cls_gts = []  # gt bboxes of this class
	cls_gt_ignore = []
	for j in range(len(gt_bboxes)):
		gt_bbox = gt_bboxes[j]
		cls_inds = (gt_labels[j] == class_id + 1)
		cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
		cls_gts.append(cls_gt)
		if gt_ignore is None:
			cls_gt_ignore.append(np.zeros(cls_gt.shape[0], dtype=np.int32))
		else:
			cls_gt_ignore.append(gt_ignore[j][cls_inds])
	return cls_dets, cls_gts, cls_gt_ignore
def kitti_eval(det_results, dataset, iou_thr=0.5):
	gt_bboxes = []
	gt_labels = []
	gt_ignore = []
	for i in range(len(dataset)):
		ann = dataset.get_ann_info(i)
		bboxes = ann['bboxes']
		labels = ann['labels']
		gt_bboxes.append(bboxes)
		gt_labels.append(labels)
		# if i>10:
		# 	break
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
config_file ='/home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_bdd_agg_fuse_st.py'

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
	torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
data_path='/backdata01/bdd'
jsonfile_name='bdd_val_3class.json'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
out_name='refer'
out_path='/home/ld/RepPoints/ld_result/bdd/epoch_29_thres0.1_nms0.5_with2'
result_record=mmcv.load(os.path.join(out_path,out_name,'det_result.pkl'))
refer_result=result_record
compute_time=0
classes = ['car','person','cyc']

results=[]
video_length=0
video_name_check=None
eval_data=[]
loc_data=[]
scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}
offset_data=[]
mask_data=[]

gt_bboxes = []
gt_labels = []
count=0
for i in range(len(dataset)):
	ann = dataset.get_ann_info(i)
	bboxes = ann['bboxes']
	labels = ann['labels']
	gt_bboxes.append(bboxes)
	gt_labels.append(labels)


num_classes = len(refer_result[0])  # positive class num
# print(num_classes)
img_name=data[0]['filename']
img=os.path.join(data_path,data[0]['filename'])
miss_refer_counts=[]
miss_agg_counts=[]
if not os.path.exists(os.path.join(os.path.join(out_path),'all_result')):
		os.mkdir(os.path.join(os.path.join(out_path),'all_result'))
for i in range(num_classes):
	miss_refer_count=0
	miss_agg_count=0
	# get gt and det bboxes of this class
	refer_dets, cls_gts, cls_gt_ignore = get_cls_results(
		refer_result, gt_bboxes, gt_labels, None, i)
	refer_more=0
	for j in range(len(cls_gts)):
		if i!=1:
			continue
		gbox=cls_gts[j]
		rbox=refer_dets[j]

		img_name=data[j]['filename']
		video_name=data[j]['video_id']
		img=os.path.join(data_path,data[j]['filename'])
		img=imshow_det_bboxes(
				img,
				gbox,
				np.ones(gbox.shape[0]).astype(np.int)*i,
				bbox_color='green',
				text_color='green',
				class_names=classes,
				show=False,
				out_file=None,out=True)
		# if i==2 and len(gbox)>0:
			# out_file=os.path.join(os.path.join(out_path),classes[i],video_name,img_name.split('/')[-1])
			# if not os.path.exists(os.path.join(os.path.join(out_path))):
			# 	os.mkdir(os.path.join(os.path.join(out_path)))
			# if not os.path.exists(os.path.join(os.path.join(out_path),classes[i])):
			# 	os.mkdir(os.path.join(os.path.join(out_path),classes[i]))
			# if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name)):
			# 	os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name))
		# 	print(out_file)
		# 	imwrite(img, out_file)
		# continue
		img=imshow_det_bboxes(
				img,
				rbox,
				np.ones(rbox.shape[0]).astype(np.int)*i,
				score_thr=0.1,
				bbox_color='red',
				text_color='red',
				class_names=classes,
				show=False,
				out_file=None,out=True)
		if i==1 and len(gbox)>0:
			out_file=os.path.join(os.path.join(out_path),'show_all',classes[i],video_name,img_name.split('/')[-1])
			if not os.path.exists(os.path.join(os.path.join(out_path),'show_all')):
				os.mkdir(os.path.join(os.path.join(out_path),'show_all'))
			if not os.path.exists(os.path.join(os.path.join(out_path),'show_all',classes[i])):
				os.mkdir(os.path.join(os.path.join(out_path),'show_all',classes[i]))
			if not os.path.exists(os.path.join(os.path.join(out_path),'show_all',classes[i],video_name)):
				os.mkdir(os.path.join(os.path.join(out_path),'show_all',classes[i],video_name))
			print(out_file)
			imwrite(img, out_file)
