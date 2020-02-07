from mmdet.apis import init_dist, init_detector, inference_detector, show_result,inference_trackor
from mmdet.core import coco_eval, results2json, wrap_fp16_model,eval_map
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import mmcv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
import matplotlib.pyplot as plt
import matplotlib
import os
import time
import ffmpeg
import json
import numpy as np
import cv2
import argparse
import os.path as osp
import shutil
import tempfile
from mmcv.image import imread, imwrite
from mmcv.utils import is_str
from enum import Enum
import torch
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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

def kernel_inv_map(vis_attr, target_point, map_h, map_w):
	pos_shift = [vis_attr['dilation'] * 0 - vis_attr['pad'],
				 vis_attr['dilation'] * 1 - vis_attr['pad'],
				 vis_attr['dilation'] * 2 - vis_attr['pad']]
	source_point = []
	for idx in range(vis_attr['filter_size']**2):
		cur_source_point = np.array([target_point[0] + pos_shift[idx //3],
									 target_point[1] + pos_shift[idx % 3]])
		if cur_source_point[0] < 0 or cur_source_point[1] < 0 \
				or cur_source_point[0] > map_h - 1 or cur_source_point[1] > map_w - 1:
			continue
		source_point.append(cur_source_point.astype('f'))
	return source_point

def offset_inv_map(source_points, offset):
	for idx, _ in enumerate(source_points):
		source_points[idx][0] += offset[2*idx]
		source_points[idx][1] += offset[2*idx+1]
	return source_points

def get_bottom_position(vis_attr, top_points, all_offset):
	map_h = all_offset[0].shape[2]
	map_w = all_offset[0].shape[3]

	for level in range(vis_attr['plot_level']):
		source_points = []
		for idx, cur_top_point in enumerate(top_points):
			cur_top_point = np.round(cur_top_point)
			if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
				or cur_top_point[0] > map_h-1 or cur_top_point[1] > map_w-1:
				continue
			cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
			cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
			cur_source_point = offset_inv_map(cur_source_point, cur_offset)
			source_points = source_points + cur_source_point
		top_points = source_points
	return source_points

def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, color=[255,0,0]):
	plot_area = vis_attr['plot_area']
	for idx, cur_source_point in enumerate(source_points):
		y = np.round((cur_source_point[0] + 0.5) * im.shape[0] / map_h).astype('i')
		x = np.round((cur_source_point[1] + 0.5) * im.shape[1] / map_w).astype('i')

		if x < 0 or y < 0 or x > im.shape[1]-1 or y > im.shape[0]-1:
			continue
		y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
		x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
		y = max(y, vis_attr['plot_area'])
		x = max(x, vis_attr['plot_area'])
		im[y-plot_area:y+plot_area+1, x-plot_area:x+plot_area+1, :] = np.tile(
			np.reshape(color, (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1)
		)
	return im

def show_dconv_offset_by_loc(count,im, all_offset, path,loc,step=[2, 2], filter_size=3,
					  dilation=1, pad=1, plot_area=1, plot_level=1,stride=8,refer=None):
	vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
				'plot_area': plot_area, 'plot_level': plot_level,'stride':stride}
	# print(all_offset[0])
	
	# a = time.time()
	map_h = all_offset[0].shape[2]
	map_w = all_offset[0].shape[3]
	# b = time.time()
	# print(b-a)
	scale=1
	
	fig=plt.figure(figsize=(10, 3))
	
	temp=0
	for (im_w,im_h) in loc:
		source_y = im_h
		source_x = im_w
		im_w=im_w/(im.shape[1] / map_w)
		im_h=im_h/(im.shape[0] / map_h)
		target_point = np.array([im_h,im_w]).astype(np.int)
		if source_y < plot_area or source_x < plot_area \
				or source_y >= im.shape[0] - plot_area or source_x >= im.shape[1] - plot_area:
			print('out of image')
			continue
		cur_im = np.copy(im)
		source_points = get_bottom_position(vis_attr, [target_point], all_offset)
		# target_offset=np.stack(source_points)
		# mean_offset=np.mean(target_offset,axis=0)
		# target_x=(mean_offset[1]* im.shape[1] / map_w).astype(np.int)
		# target_y=(mean_offset[0]* im.shape[0] / map_h).astype(np.int)

		cur_im = plot_according_to_point(vis_attr, cur_im, source_points, map_h, map_w)
		# cur_im[source_y-4:source_y+4+1, source_x-4:source_x+4+1, :] = \
		# 	np.tile(np.reshape([255, 255, 0], (1, 1, 3)), (2*4+1, 2*4+1, 1))
		# # cur_im[target_y-4:target_y+4+1, target_x-4:target_x+4+1, :] = \
						# np.tile(np.reshape([0, 0, 255], (1, 1, 3)), (2*4+1, 2*4+1, 1))
		# valid_num=np.sum(np.where(cur_im[:,:,0]==255,1,0)*np.where(cur_im[:,:,1]==0,1,0)*np.where(cur_im[:,:,2]==0,1,0))
		# plt.text(0,0,'valid_offset:%d,offset_ratio:%.2f'%(valid_num,valid_num/6561),fontdict={'size': 10, 'color':  'black'})
		print('showing',im_h,im_w)
		plt.axis("off")
		plt.imshow(cur_im)
		fig.savefig(path + '/' + str(count) + '_' + str(temp) + '.jpg',dpi=150)
		#path + '/' + str(count) + '_' + str(temp) + '.jpg'
		plt.clf()
		temp+=1
	plt.close('all')

config_file ='/home/hrb/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_mt.py'
cfg = mmcv.Config.fromfile(config_file)
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
out_name='refer'
out_path = '/home/hrb/baseline'

result_record = mmcv.load(os.path.join(out_path,out_name,'result/det_result.pkl'))

gt_bboxes = []
gt_labels = []
for i in range(len(dataset)):
	ann = dataset.get_ann_info(i)
	bboxes = ann['bboxes']
	labels = ann['labels']
	gt_bboxes.append(bboxes)
	gt_labels.append(labels)

refer_result = result_record
num_classes = len(refer_result[0])
# print(num_classes)  
img_name=data[0]['filename']
img=os.path.join(data_path,data[0]['filename'])

# scale = [8,16,32,64,128]
for m in range(4,0,-1):
	offset = np.load("/home/hrb/offset/dcn_offset"+str(m)+".npy")
	offset = offset.tolist() 
	offset = np.vstack(offset)
	save_path = os.path.join(out_path,out_name,'offset','offset_'+str(m))
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	else:
		shutil.rmtree(save_path)
		os.mkdir(save_path)
	count = 0
	for i in range(num_classes):
		refer_dets, cls_gts, cls_gt_ignore = get_cls_results(
			refer_result, gt_bboxes, gt_labels, None, i)
		
		for j in range(len(cls_gts)):
			gbox=cls_gts[j]
			rbox=refer_dets[j]
			gr_iou=bbox_overlaps(gbox,rbox)
			ground_index=(1-((gr_iou>0.5).astype(np.float).sum(axis=1)>0)).astype(np.bool)
			miss_gound=gbox[ground_index]
			if ground_index.astype(np.float).sum()>0:
				img_name=data[j]['filename']
				video_name=data[j]['video_id']
				img=os.path.join(data_path,data[j]['filename'])
				if not os.path.exists(os.path.join(os.path.join(out_path,out_name),'image',video_name)):
					os.mkdir(os.path.join(os.path.join(out_path,out_name),'image',video_name))
				print(img_name)
				img = imshow_det_bboxes(
							img,
							miss_gound,
							np.ones(miss_gound.shape[0]).astype(np.int)*i,
							class_names=classes,
							show=False,
							# out_file=os.path.join(os.path.join(out_path,out_name),'image',video_name,img_name.split('/')[-1]))
							out_file=None)
				loc = []
				
				if miss_gound.shape[0] > 0:
					for i in range(miss_gound.shape[0]):
						source_x = (miss_gound[i][0] + miss_gound[i][2]) / 2
						source_y = (miss_gound[i][1] + miss_gound[i][3]) / 2
						loc.append([source_x, source_y])
					all_offset = offset[j]
					all_offset = all_offset[np.newaxis,:]

					show_dconv_offset_by_loc(count,img,[all_offset],save_path,loc,plot_level=1,plot_area=3)
					
					count+=1
					# print(count)
# print(offset.shape)