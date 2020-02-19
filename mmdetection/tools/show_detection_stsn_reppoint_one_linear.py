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

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json
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

def get_cls_results(det_results, gt_bboxes, gt_labels, class_id,loc_results):
	"""Get det results and gt information of a certain class."""
	cls_dets = [det[class_id]
				for det in det_results]  # det bboxes of this class
	cls_locs = [loc[class_id]
				for loc in loc_results]  # det bboxes of this class
	cls_gts = []  # gt bboxes of this class

	for j in range(len(gt_bboxes)):
		gt_bbox = gt_bboxes[j]
		cls_inds = (gt_labels[j] == class_id + 1)
		cls_gt = gt_bbox[cls_inds, :] if gt_bbox.shape[0] > 0 else gt_bbox
		cls_gts.append(cls_gt)
	return cls_dets, cls_gts,cls_locs
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
config_file ='/home/ld/RepPoints/configs/stsn_do3.py'

cfg = mmcv.Config.fromfile(config_file)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
	torch.backends.cudnn.benchmark = True
cfg.model.pretrained = None
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
data_path='/backdata01/'
jsonfile_name='kitti_bdd_waymo_2class_val_13.json'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
compute_time=0
classes = ['Car','Person']
out_name='agg'
out_path='/home/ld/RepPoints/ld_result/stsn_one_linear_do3/bilinear/epoch_30_thres0.3_nms0.5_support_5/'+out_name
results=[]
video_length=0
video_name_check=None
result_record=[]
eval_data=[]
loc_data=[]
# scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}
offset_data=[]
refer_record=mmcv.load(os.path.join('/home/ld/RepPoints/ld_result/stsn_one_linear_do3/bilinear/epoch_30_thres0.3_nms0.5_support_5/refer/det_result.pkl'))
result_record=mmcv.load(os.path.join(out_path,'det_result.pkl'))
loc_result=mmcv.load(os.path.join(out_path,'loc_result.pkl'))
img_record=mmcv.load(os.path.join(out_path,'images.pkl'))
# load and test on kitti
# out_path='/home/ld/RepPoints/final/fuse_c_result/epoch9_thres0.1_nms0.3_with10'
# result_record=mmcv.load(os.path.join(out_path,out_name,'det_result.pkl'))
# print('evaluating result of ', out_name)
# kitti_eval(result_record, dataset)
# exit()
gt_bboxes = []
gt_labels = []
for i in range(len(dataset)):
	ann = dataset.get_ann_info(i)
	bboxes = ann['bboxes']
	labels = ann['labels']
	gt_bboxes.append(bboxes)
	gt_labels.append(labels)


num_classes = len(classes)
img_name=data[0]['filename']
img=os.path.join(data_path,data[0]['filename'])
# show the first image if needed
# show_result(img, result_record[0], classes, show=True,out_file=os.path.join(os.path.join(out_path),img_name.split('/')[-1]))

# show the ground truth label if needed
# imshow_det_bboxes(
# 		img,
# 		np.vstack(data[0]['ann']['bboxes']),
# 		np.vstack(data[0]['ann']['labels']).astype(np.int).squeeze()-1,
# 		class_names=classes,
# 		show=True,
# 		out_file=os.path.join(os.path.join(out_path),img_name.split('/')[-1]))
# reppoints=mmcv.load('/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9_thres0.1_nms0.5_with2/epoch_9_thres0.1_nms0.5_with2/refer/reppoints.pkl')



index=[[] for i in range(3)]
for i in range(num_classes):
	if not os.path.exists(os.path.join(os.path.join(out_path),classes[i])):
		os.mkdir(os.path.join(os.path.join(out_path),classes[i]))
	not_det_count=0
	wrong_det_count=0
	# get gt and det bboxes of this class
	dets, cls_gts,locs = get_cls_results(
		result_record, gt_bboxes, gt_labels, i,loc_result)
	refer_dets, _,_ = get_cls_results(
		refer_record, gt_bboxes, gt_labels, i,loc_result)
	for j in range(len(cls_gts)):
		print(data[j]['filename'])
		gbox=cls_gts[j]
		pbox=dets[j]
		rbox=refer_dets[j]
		ploc=locs[j]
		gp_iou=bbox_overlaps(gbox,pbox)
		gr_iou=bbox_overlaps(gbox,rbox)
		#iou matrix: dim=0 ground, dim=1 prediction
		#not detected: if there is no prediction bbox have a iou more than 0.5 to the ground truth
		not_detected_index=((gp_iou>0.5).astype(np.float).sum(axis=1)>0)==0
		refer_not_detected_index=((gr_iou>0.5).astype(np.float).sum(axis=1)>0)==0
		addition_index=refer_not_detected_index*(1-not_detected_index)>0
		#not detected bbox
		addition=gbox[addition_index]
		print(data[j]['filename'],img_record[j])

		#wrong detected: if no ground truth bbox having a iou more than 0.5 to the prediction
		# wrong_detect_index=(gp_iou>0.5).astype(np.float).sum(axis=0)==0
		# wrong_detected=pbox[wrong_detect_index]
		# wrong_loc=ploc[wrong_detect_index]

		if len(addition)>0:
			addition_loc=[]
			for m in range(len(addition)):
				addition_loc.append([(addition[m][0]+addition[m][2])//2,(addition[m][1]+addition[m][3])//2,0,1])
			img_name=data[j]['filename']
			video_name=data[j]['video_id']
			img=os.path.join(data_path,data[j]['filename'])
			file_name=os.path.join(os.path.join(out_path),classes[i],video_name,'addition',img_name.split('/')[-1])
			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name)):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name))
			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name,'addition')):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name,'addition'))
			imshow_det_bboxes(
					img,
					addition,
					np.ones(addition.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					bbox_color='green',
					text_color='green',
					out_file=file_name)

			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name,'add_support1')):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name,'add_support1'))
			img=img_record[j][1]
			add_support1_name=os.path.join(os.path.join(out_path),classes[i],video_name,'add_support1',img.split('/')[-1])
			imshow_det_bboxes(
					img,
					addition,
					np.ones(addition.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					bbox_color='green',
					text_color='green',
					out_file=add_support1_name)

			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name,'add_support2')):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name,'add_support2'))
			img=img_record[j][2]
			add_support2_name=os.path.join(os.path.join(out_path),classes[i],video_name,'add_support2',img.split('/')[-1])
			imshow_det_bboxes(
					img,
					addition,
					np.ones(addition.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					bbox_color='green',
					text_color='green',
					out_file=add_support2_name)
			# if data[j]['filename'].split('/')[-1]=='000204.png':
			# 	print(file_name,add_support1_name,add_support2_name)
			# 	exit()
			# continue
			mmcv.dump([addition_loc,j,file_name,add_support1_name,add_support2_name,data[j]['filename'].split('/')[2]], os.path.join(os.path.join(out_path),classes[i],video_name,'addition',img_name.split('/')[-1].split('.')[0]+'.pkl'))

		#not detected: if there is no prediction bbox have a iou more than 0.5 to the ground truth
		not_detected_index=((gp_iou>0.5).astype(np.float).sum(axis=1)>0)==0
		#not detected bbox
		not_detected=gbox[not_detected_index]
		
		if len(not_detected)>0:
			no_loc=[]
			for m in range(len(not_detected)):
				no_loc.append([(not_detected[m][0]+not_detected[m][2])//2,(not_detected[m][1]+not_detected[m][3])//2,0,1])
			img_name=data[j]['filename']
			video_name=data[j]['video_id']
			img=os.path.join(data_path,data[j]['filename'])
			file_name=os.path.join(os.path.join(out_path),classes[i],video_name,'not_detected',img_name.split('/')[-1])
			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name)):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name))
			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name,'not_detected')):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name,'not_detected'))
			imshow_det_bboxes(
					img,
					not_detected,
					np.ones(not_detected.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					bbox_color='green',
					text_color='green',
					out_file=file_name)

			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name,'support1')):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name,'support1'))
			img=img_record[j][1]
			support1_name=os.path.join(os.path.join(out_path),classes[i],video_name,'support1',img.split('/')[-1])
			imshow_det_bboxes(
					img,
					not_detected,
					np.ones(not_detected.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					bbox_color='green',
					text_color='green',
					out_file=support1_name)

			if not os.path.exists(os.path.join(os.path.join(out_path),classes[i],video_name,'support2')):
				os.mkdir(os.path.join(os.path.join(out_path),classes[i],video_name,'support2'))
			img=img_record[j][2]
			support2_name=os.path.join(os.path.join(out_path),classes[i],video_name,'support2',img.split('/')[-1])
			imshow_det_bboxes(
					img,
					not_detected,
					np.ones(not_detected.shape[0]).astype(np.int)*i,
					class_names=classes,
					show=False,
					bbox_color='green',
					text_color='green',
					out_file=support2_name)
			print(data[j]['filename'].split('/')[2])
			mmcv.dump([no_loc,j,file_name,support1_name,support2_name,data[j]['filename'].split('/')[2]], os.path.join(os.path.join(out_path),classes[i],video_name,'not_detected',img_name.split('/')[-1].split('.')[0]+'.pkl'))
	