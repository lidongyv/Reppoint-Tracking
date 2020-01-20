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
checkpoint_file='/home/ld/RepPoints/final/fuse_cluster/epoch_9.pth'
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

out_name=['refer','agg']
for i in range(10):
	out_name.append('frame_'+str(i+1))
out_path='/home/ld/RepPoints/final/fuse_cluster/epoch9_thres0.1_nms0.3_with3'
if not os.path.exists(out_path):
	os.mkdir(out_path)
	for i in range(12):
		os.mkdir(os.path.join(out_path,out_name[i]))
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

#load and test
# outputs=mmcv.load(out_path+'/fuse_result.pkl')
# kitti_eval(outputs, dataset)
# exit()


# for i in range(12):
# 	result_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'det_result.pkl'))
# for i in range(12):
# 	print('evaluating result of ', out_name[i])
# 	kitti_eval(result_record[i], dataset)
# exit()


# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

model.CLASSES = dataset.CLASSES

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
	if video_length <11:
		for j in range(1,11,1):
			img_list.append(os.path.join(data_path,data[i+j]['filename']))
		
	else:
		for j in range(-1,-11,-1):
			img_list.append(os.path.join(data_path,data[i+j]['filename']))
	# img1=os.path.join(data_path,data[i+10]['filename'])
	result = inference_trackor(model, img_list)
	for j in range(12):
		bbox_result=result[j][0]
		loc_result=result[j][1].long()
		result_record[j].append(bbox_result)
		loc_data[j].append(loc_result)
		#four value and one score
		bboxes = np.vstack(bbox_result)
		scores = bboxes[:, -1]
		inds = scores > 0
		scores=bboxes[inds, :][:,4:]
		bboxes = bboxes[inds, :][:,:4]
		offset_loc=[]
		mask_loc=[]
		if j>1:
			offset=model.agg.offset[j-2]
			mask=model.agg.mask[j-2]
			for m in range(len(loc_result)):
				# print(offset[scale[str(loc_result[m,2].item())]].shape)
				# print(loc_result[m,2])
				# print(loc_result[m,0]//loc_result[m,2])
				offset_loc.append(offset[scale[str(loc_result[m,2].item())]][0,:,loc_result[m,1]//loc_result[m,2],loc_result[m,0]//loc_result[m,2]].data.cpu())
				mask_loc.append(mask[scale[str(loc_result[m,2].item())]][0,:,loc_result[m,1]//loc_result[m,2],loc_result[m,0]//loc_result[m,2]].data.cpu())
			# print(j-2)
			# print(len(offset_loc),offset_loc[0].shape)
			offset_data[j-2].append(offset_loc)
			mask_data[j-2].append(mask_loc)
		# show_result(img, result, model.CLASSES, show=False,out_file=os.path.join(out_path,video_name,img_name))

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
	# if i >10:
	# 	break
for i in range(12):
	mmcv.dump(result_record[i], os.path.join(out_path,out_name[i],'det_result.pkl'))
	mmcv.dump(loc_data[i], os.path.join(out_path,out_name[i],'loc_result.pkl'))
	mmcv.dump(eval_data[i], os.path.join(out_path,out_name[i],'track.pkl'))
	if i>1:
		mmcv.dump(offset_data[i-2], os.path.join(out_path,out_name[i],'offset.pkl'))
		mmcv.dump(mask_data[i-2], os.path.join(out_path,out_name[i],'mask.pkl'))

# for i in range(12):
# 	result_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'det_result.pkl'))
for i in range(12):
	print('evaluating result of ', out_name[i])
	kitti_eval(result_record[i], dataset)
# with open(os.path.join('./result/','retina_'+jsonfile_name),'w+',encoding='utf-8') as f:
# 	data=json.dump(eval_data,f)
# mmcv.dump(outputs, args.out)