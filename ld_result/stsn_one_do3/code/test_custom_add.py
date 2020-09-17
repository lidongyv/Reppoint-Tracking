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
config_file ='/home/ld/RepPoints/configs/stsn_do3.py'
checkpoint_file='/home/ld/RepPoints/ld_result/stsn_one_do3/epoch_30.pth'
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
support_count=2
support_num=5
out_name='refer'
out_path='/home/ld/RepPoints/ld_result/stsn_one_do3/epoch_30_thres0.3_nms0.5_support_'+str(support_num)+'add'

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
reppooints_data=[[] for i in range(5)]
offset_data=[[] for i in range(5)]
offset_data=[copy.deepcopy(offset_data),copy.deepcopy(offset_data),copy.deepcopy(offset_data),copy.deepcopy(offset_data),copy.deepcopy(offset_data)]


img_record=[]
scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}


# # load and test
# offset=mmcv.load('/home/ld/RepPoints/ld_result/stsn_one_do3/epoch_30_thres0.3_nms0.5_support_5/refer/reppoints.pkl')
# print(offset[0][0].shape)
# result_record=mmcv.load(os.path.join(out_path,'refer/det_result.pkl'))
# print('evaluating result of support', )

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


	if data[i-1]['video_id']==video_name and data[i-support_num]['video_id']==video_name:
		img_list.append(os.path.join(data_path,data[i-1]['filename']))
		img_list.append(os.path.join(data_path,data[i-support_num]['filename']))
	else:
		img_list.append(os.path.join(data_path,data[i+1]['filename']))
		img_list.append(os.path.join(data_path,data[i+support_num]['filename']))
	if i<len(data)-6:
		if data[i+1]['video_id']==video_name and data[i+support_num]['video_id']==video_name:
			img_list.append(os.path.join(data_path,data[i+1]['filename']))
			img_list.append(os.path.join(data_path,data[i+support_num]['filename']))
		else:
			img_list.append(os.path.join(data_path,data[i-1]['filename']))
			img_list.append(os.path.join(data_path,data[i-support_num]['filename']))
	else:
		img_list.append(os.path.join(data_path,data[i-1]['filename']))
		img_list.append(os.path.join(data_path,data[i-support_num]['filename']))
	img_record.append(img_list)
	result = inference_trackor(model, img_list)
	reppoint_t=model.bbox_head.reppoints
	offset_t=model.bbox_head.offset
	bbox_result=result[0]
	loc_result=result[1]
	result_record.append(bbox_result)
	loc_data.append(loc_result)
	for m in range(len(reppoint_t)):
		reppooints_data[m].append(reppoint_t[m])
		print('reppoint',reppoint_t[m].shape)
	for m in range(len(offset_t)):
		for n in range(len(offset_t[m])):
			offset_data[n][m].append(offset_t[m][n])



mmcv.dump(img_record, os.path.join(out_path,out_name,'images.pkl'))
mmcv.dump(offset_data, os.path.join(out_path,out_name,'offsets.pkl'))
mmcv.dump(result_record, os.path.join(out_path,out_name,'det_result.pkl'))
mmcv.dump(loc_data, os.path.join(out_path,out_name,'loc_result.pkl'))
mmcv.dump(reppooints_data, os.path.join(out_path,out_name,'reppoints.pkl'))


print('evaluating result of ', out_name)
kitti_eval(result_record, dataset)
