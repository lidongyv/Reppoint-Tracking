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
config_file ='/home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_agg_fuse_st.py'

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
	torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
data_path='/backdata01/KITTI/kitti/tracking'
jsonfile_name='kitti_val_3class.json'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)

compute_time=0
classes = ['Vehicle','Pedestrian','Cyclist']
out_name=['refer','agg']
for i in range(10):
	out_name.append('frame_'+str(i+1))
# out_path='/home/ld/RepPoints/analyze/fuse_result/epoch13 thres0.1'
out_path='/home/ld/RepPoints/final/fuse_c_result/epoch9_thres0.1_nms0.3_with2'
results=[]
video_length=0
video_name_check=None
result_record=[]
for i in range(12):
	result_record.append([])
eval_data=[]
for i in range(12):
	eval_data.append([])
loc_data=[]
for i in range(12):
	loc_data.append([])
scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}
offset_data=[]
for i in range(10):
	offset_data.append([])
mask_data=[]
for i in range(10):
	mask_data.append([])
for i in range(12):
	result_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'det_result.pkl'))
# out_path='/home/ld/RepPoints/final/fuse_c_result/epoch9_thres0.1_nms0.3_with10'
# for i in range(12):
# 	result_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'det_result.pkl'))
# for i in range(12):
# 	print('evaluating result of ', out_name[i])
# 	kitti_eval(result_record[i], dataset)
# exit()
gt_bboxes = []
gt_labels = []
for i in range(len(dataset)):
	ann = dataset.get_ann_info(i)
	bboxes = ann['bboxes']
	labels = ann['labels']
	gt_bboxes.append(bboxes)
	gt_labels.append(labels)
	# print(len(bboxes))
#a list of list, [[cls1_det, cls2_det, ...], ...]
refer_result=result_record[0]
agg_result=result_record[1]

num_classes = len(refer_result[0])  # positive class num
# print(num_classes)
img_name=data[0]['filename']
img=os.path.join(data_path,data[0]['filename'])
# show_result(img, refer_result[0], classes, show=True,out_file=os.path.join(os.path.join(out_path),img_name.split('/')[-1]))
# exit()
# imshow_det_bboxes(
# 		img,
# 		np.vstack(data[0]['ann']['bboxes']),
# 		np.vstack(data[0]['ann']['labels']).astype(np.int).squeeze()-1,
# 		class_names=classes,
# 		show=True,
# 		out_file=os.path.join(os.path.join(out_path),img_name.split('/')[-1]))
# exit()
miss_refer_counts=[]
miss_agg_counts=[]
if not os.path.exists(os.path.join(os.path.join(out_path),'refer_loss')):
	os.mkdir(os.path.join(os.path.join(out_path),'refer_loss'))
if not os.path.exists(os.path.join(os.path.join(out_path),'agg_loss')):
	os.mkdir(os.path.join(os.path.join(out_path),'agg_loss'))
if not os.path.exists(os.path.join(os.path.join(out_path),'baseline')):
		os.mkdir(os.path.join(os.path.join(out_path),'baseline'))
for i in range(num_classes):
	miss_refer_count=0
	miss_agg_count=0
	# get gt and det bboxes of this class
	refer_dets, cls_gts, cls_gt_ignore = get_cls_results(
		refer_result, gt_bboxes, gt_labels, None, i)
	agg_dets, cls_gts, cls_gt_ignore = get_cls_results(
		agg_result, gt_bboxes, gt_labels, None, i)
	refer_more=0
	agg_more=0
	# for j in range(len(cls_gts)):
	# 	img_name=data[j]['filename']
	# 	img=os.path.join(data_path,data[0]['filename'])
	# 	# show_result(img, refer_result[j], classes, show=True,out_file=os.path.join(os.path.join(out_path),img_name.split('/')[-1]))
	# 	if len(refer_dets[j])>len(agg_dets[j]):
	# 		refer_more+=1
	# 	else:
	# 		agg_more+=1

	# img_name=data[0]['filename']
	# img=os.path.join(data_path,data[0]['filename'])
	# imshow_det_bboxes(
	# 		img,
	# 		np.vstack(refer_result[0][i]),
	# 		np.ones(len(refer_result[0][i])).astype(np.int)*i,
	# 		class_names=classes,
	# 		show=True,
	# 		out_file=os.path.join(os.path.join(out_path),img_name.split('/')[-1]))

	for j in range(len(cls_gts)):
		gbox=cls_gts[j]
		rbox=refer_dets[j]
		abox=agg_dets[j]
		gr_iou=bbox_overlaps(gbox,rbox)
		ground_index=(1-((gr_iou>0.5).astype(np.float).sum(axis=1)>0)).astype(np.bool)
		miss_gound=gbox[ground_index]

	
		tp_refer_iou=(gr_iou>0.5).astype(np.float)
		refer_index=(tp_refer_iou.sum(axis=0)>0)
		tp_refer=rbox[refer_index]
		ga_iou=bbox_overlaps(gbox,abox)
		tp_agg_iou=(ga_iou>0.5).astype(np.float)
		agg_index=(tp_agg_iou.sum(axis=0)>0)
		tp_agg=abox[agg_index]
	
		ra_iou=bbox_overlaps(tp_refer,tp_agg)
		# print(ra_iou)
		ra_index=(1-((ra_iou>0.5).astype(np.float).sum(axis=1)>0)).astype(np.bool)
		# print(ra_index)
		miss_refer=rbox[refer_index][ra_index]
		# print(miss_refer)
		ar_index=(1-((ra_iou>0.5).astype(np.float).sum(axis=0)>0)).astype(np.bool)
		# print(ra_index)
		miss_agg=abox[agg_index][ar_index]

		if ground_index.astype(np.float).sum()>0:
			img_name=data[j]['filename']
			video_name=data[j]['video_id']
			img=os.path.join(data_path,data[j]['filename'])
			if not os.path.exists(os.path.join(os.path.join(out_path),'baseline',video_name)):
				os.mkdir(os.path.join(os.path.join(out_path),'baseline',video_name))
			print(img_name)
			imshow_det_bboxes(
					img,
					miss_gound,
					np.ones(miss_gound.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					out_file=os.path.join(os.path.join(out_path),'baseline',video_name,img_name.split('/')[-1]))
		if ra_index.astype(np.float).sum()>0:
			miss_refer_count+=ra_index.astype(np.float).sum()
			img_name=data[j]['filename']
			video_name=data[j]['video_id']
			img=os.path.join(data_path,data[j]['filename'])
			if not os.path.exists(os.path.join(os.path.join(out_path),'refer_loss',video_name)):
				os.mkdir(os.path.join(os.path.join(out_path),'refer_loss',video_name))
			print(img_name,ra_index.astype(np.float).sum())
			# imshow_det_bboxes(
			# 		img,
			# 		tp_refer,
			# 		np.ones(tp_refer.shape[0]).astype(np.int)*i,
			# 		class_names=classes,
			# 		show=True,
			# 		out_file=os.path.join(os.path.join(out_path),video_name,img_name.split('/')[-1]))
			# imshow_det_bboxes(
			# 		img,
			# 		tp_agg,
			# 		np.ones(tp_agg.shape[0]).astype(np.int)*i,
			# 		class_names=classes,
			# 		show=True,
			# 		out_file=os.path.join(os.path.join(out_path),video_name,img_name.split('/')[-1]))

			imshow_det_bboxes(
					img,
					miss_refer,
					np.ones(miss_refer.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					out_file=os.path.join(os.path.join(out_path),'refer_loss',video_name,img_name.split('/')[-1]))
		if ar_index.astype(np.float).sum()>0:
			miss_agg_count+=ar_index.astype(np.float).sum()
			img_name=data[j]['filename']
			video_name=data[j]['video_id']
			img=os.path.join(data_path,data[j]['filename'])
			if not os.path.exists(os.path.join(os.path.join(out_path),'agg_loss',video_name)):
				os.mkdir(os.path.join(os.path.join(out_path),'agg_loss',video_name))
			print(img_name,ar_index.astype(np.float).sum())
			imshow_det_bboxes(
					img,
					miss_agg,
					np.ones(miss_agg.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					out_file=os.path.join(os.path.join(out_path),'agg_loss',video_name,img_name.split('/')[-1]))
			# exit()
		# print(gbox,rbox,abox)
	miss_refer_counts.append(miss_refer_count)
	miss_agg_counts.append(miss_agg_count)
	
# print(miss_refer_counts)
# print(miss_agg_counts)
# [188.0, 96.0, 3.0]
# [139.0, 81.0, 5.0]
	# exit()
	# print(cls_gts[0])
	# print(refer_dets[0])
	# print(agg_dets[0])
