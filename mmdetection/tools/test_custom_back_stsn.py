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
checkpoint_file='/home/ld/RepPoints/debug/reppoint_stsn/epoch_13.pth'
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
out_name=['refer','refer_trans','agg','agg_refer']

for i in range(support_count):
	out_name.append('frame_'+str(i+1))
out_path='/home/ld/RepPoints/debug/reppoint_stsn/epoch13_thres0.1_nms0.5_with2_refine'
if not os.path.exists(out_path):
	os.mkdir(out_path)
	for i in range(len(out_name)):
		os.mkdir(os.path.join(out_path,out_name[i]))
results=[]
video_length=0
video_name_check=None
result_record=[]
for i in range(len(out_name)):
	result_record.append([])
eval_data=[]
for i in range(len(out_name)):
	eval_data.append([])
loc_data=[]
for i in range(len(out_name)):
	loc_data.append([])
scale=[8,16,32,64,128]
scale={'8':0,'16':1,'32':2,'64':3,'128':4}
offset_data=[]
for i in range(support_count):
	offset_data.append([])
offset_data=[]
for i in range(support_count):
	offset_data.append([])

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
	if i+2>=len(data):
		img_list.append(os.path.join(data_path,data[i]['filename']))
	else:
		if data[i+2]['video_id']==video_name:
			img_list.append(os.path.join(data_path,data[i+2]['filename']))
		else:
			img_list.append(os.path.join(data_path,data[i]['filename']))

	result = inference_trackor(model, img_list)
	offset=model.agg.offset
	for j in range(len(out_name)):

		bbox_result=result[j][0]
		loc_result=result[j][1]
		result_record[j].append(bbox_result)
		loc_data[j].append(loc_result)
		loc_result=loc_result.long()
		#four value and one score
		bboxes = np.vstack(bbox_result)
		scores = bboxes[:, -1]
		inds = scores > 0
		scores=bboxes[inds, :][:,4:]
		bboxes = bboxes[inds, :][:,:4]

		labels = [
			np.full(bbox.shape[0], i, dtype=np.int32)
			for i, bbox in enumerate(bbox_result)
		]
		labels = np.concatenate(labels)
		labels = labels[inds]
		frame_data={"video_id":frame['video_id'],"filename":os.path.join(frame['filename']), \
			"ann":{"bboxes":bboxes.tolist(),"labels":labels.tolist(), \
				"track_id":labels.tolist(),'score':scores.tolist()}}
		eval_data[j].append(frame_data)

		offset_loc=[]
		mask_loc=[]
		if j>=4:
			print(len(model.agg.offset))
			offset=model.agg.offset[j-4]
			for m in range(len(loc_result)):
				offset_loc.append(offset[scale[str(loc_result[m,2].item())]][0,:,loc_result[m,1]//loc_result[m,2],loc_result[m,0]//loc_result[m,2]].data.cpu())
			offset_data[j-4].append(offset_loc)
for i in range(len(out_name)):
	mmcv.dump(result_record[i], os.path.join(out_path,out_name[i],'det_result.pkl'))
	mmcv.dump(loc_data[i], os.path.join(out_path,out_name[i],'loc_result.pkl'))
	# mmcv.dump(eval_data[i], os.path.join(out_path,out_name[i],'track.pkl'))
	# if i>1:
	# 	mmcv.dump(offset_data[i-2], os.path.join(out_path,out_name[i],'offset.pkl'))
for i in range(support_count):
	
	mmcv.dump(offset_data[i-2], os.path.join(out_path,out_name[i+4],'offset.pkl'))
for i in range(len(out_name)):
	print('evaluating result of ', out_name[i])
	kitti_eval(result_record[i], dataset)

