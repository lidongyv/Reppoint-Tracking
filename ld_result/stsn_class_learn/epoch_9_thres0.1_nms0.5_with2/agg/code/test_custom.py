# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-25 19:24:06  
#@Last Modified by: Lidong Yu  
import copy

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
checkpoint_file='/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9.pth'
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
support_count=2
out_name='agg'
out_path='/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9_thres0.1_nms0.5_with2'
if not os.path.exists(out_path):
	os.mkdir(out_path)
if not os.path.exists(os.path.join(out_path,out_name)):
	os.mkdir(os.path.join(out_path,out_name))
results=[]
video_length=0
video_name_check=None
result_record=[]

eval_data=[]

loc_data=[]
offset_data=[[] for i in range(5)]
offset_data=[copy.deepcopy(offset_data),copy.deepcopy(offset_data)]
reppoint_data=[[] for i in range(5)]
img_record=[]
scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}


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
for i,(frame) in enumerate(data):
	print(i,'in',len(data))
	video_name=frame['video_id']
	if video_name_check is None:
		video_name_check=video_name
	else:
		if video_name_check==video_name:
			video_length+=1
		else:
			video_name_check=video_name
			video_length=0
	print('video_name',video_name,'image_name',frame['filename'],'video_length',video_length)
	img_name=frame['filename']
	# img = mmcv.imread(os.path.join(data_path,img_name))
	img=os.path.join(data_path,img_name)
	img_list=[img]


	if data[i-2]['video_id']==video_name:
		img_list.append(os.path.join(data_path,data[i-2]['filename']))
	else:
		img_list.append(os.path.join(data_path,data[i]['filename']))
		print('start')
	if i+2>=len(data):
		img_list.append(os.path.join(data_path,data[i]['filename']))
	else:
		if data[i+2]['video_id']==video_name:
			img_list.append(os.path.join(data_path,data[i+2]['filename']))
		else:
			img_list.append(os.path.join(data_path,data[i]['filename']))
			print('end')
	img_record.append(img_list)
	result = inference_trackor(model, img_list)
	offset_t=model.bbox_head.offset
	reppoint_t=model.bbox_head.reppoints
	bbox_result=result[0]
	loc_result=result[1]
	result_record.append(bbox_result)
	loc_data.append(loc_result)
	for m in range(len(reppoint_t)):
		reppoint_data[m].append(reppoint_t[m])
	for m in range(len(offset_t)):
		for n in range(len(offset_t[m])):
			offset_data[n][m].append(offset_t[m][n])

	# loc_result=loc_result.long()
	# #four value and one score
	# bboxes = np.vstack(bbox_result)
	# scores = bboxes[:, -1]
	# inds = scores > 0
	# scores=bboxes[inds, :][:,4:]
	# bboxes = bboxes[inds, :][:,:4]

	# labels = [
	# 	np.full(bbox.shape[0], i, dtype=np.int32)
	# 	for i, bbox in enumerate(bbox_result)
	# ]
	# labels = np.concatenate(labels)
	# labels = labels[inds]
	# frame_data={"video_id":frame['video_id'],"filename":os.path.join(frame['filename']), \
	# 	"ann":{"bboxes":bboxes.tolist(),"labels":labels.tolist(), \
	# 		"track_id":labels.tolist(),'score':scores.tolist()}}
	# eval_data.append(frame_data)
mmcv.dump(img_record, os.path.join(out_path,out_name,'images.pkl'))
mmcv.dump(offset_data, os.path.join(out_path,out_name,'offset.pkl'))
mmcv.dump(reppoint_data, os.path.join(out_path,out_name,'reppoints.pkl'))
# exit()
mmcv.dump(result_record, os.path.join(out_path,out_name,'det_result.pkl'))
mmcv.dump(loc_data, os.path.join(out_path,out_name,'loc_result.pkl'))


print('evaluating result of ', out_name)
kitti_eval(result_record, dataset)

