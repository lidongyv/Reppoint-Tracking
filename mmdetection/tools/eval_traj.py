# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-25 19:24:06  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-25 19:24:06

from mmdet.apis import init_detector, inference_trajectory, show_result
import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import cv2
import torch
# config_file = '/home/ld/RepPoints/mmdetection/configs/retinanet_r101_fpn_1x.py'
# checkpoint_file = '/home/ld/RepPoints/trained/retinanet_r101_fpn_1x_20181129-f016f384.pth'
config_file ='/home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti.py'
checkpoint_file='/home/ld/RepPoints/work_dirs/reppoints_moment_r101_dcn_fpn_kitti/epoch_30.pth'
# config_file ='/home/ld/RepPoints/configs/retinanet_r101_fpn_kitti.py'
# checkpoint_file='/home/ld/RepPoints/work_dirs/retinanet_r101_fpn_kitti/epoch_50.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
data_path='/backdata01/KITTI/kitti/tracking'
# ['calib', 'disparity', 'image_02', 'label_02', 'splits', 'velodyne', 'velodyne_img', 
# 'Flow', 'Inv_Flow', 'trackinglabel_02', 'kitti.json', 'shuffle data', 'kitti_train.json'
#  'kitti_val.json']
jsonfile_name='kitti_val.json'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
compute_time=0
eval_data=[]
out_path='/home/ld/RepPoints/out/reppoint'
current_tr=None
for i,(frame) in enumerate(data):
	print(i,'in',len(data))
	video_name=frame['video_id']
	print('video_name',video_name,'image_name',frame['filename'])
	img_name=frame['filename']
	img = mmcv.imread(os.path.join(data_path,img_name))
	if current_tr!=video_name:
		trajectory=torch.load('/backdata01/KITTI/kitti/trajectory/'+video_name+'.pkl')
		video_length=0
	result = inference_trajectory(model, img)
	# if not os.path.exists(os.path.join(out_path,video_name)):
	# 	os.mkdir(os.path.join(out_path,video_name))
	# show_result(img, result, model.CLASSES, show=False,out_file=os.path.join(out_path,video_name,img_name))
	if isinstance(result, tuple):
		bbox_result, segm_result = result
	else:
		bbox_result, segm_result = result, None
	#four value and one score
	bboxes = np.vstack(bbox_result)
	scores = bboxes[:, -1]
	inds = scores > 0.5
	bboxes = bboxes[inds, :][:,:4]
	
	# draw bounding boxes
	labels = [
		np.full(bbox.shape[0], i, dtype=np.int32)
		for i, bbox in enumerate(bbox_result)
	]
	labels = np.concatenate(labels)
	labels = labels[inds]
	frame_data={"video_id":frame['video_id'],"filename":os.path.join(frame['filename']), \
		"ann":{"bboxes":bboxes.tolist(),"labels":labels.tolist(), \
			"track_id":labels.tolist()},"flow_name":frame['flow_name'],"inv_flow_name":frame['inv_flow_name']}
	eval_data.append(frame_data)
	video_length+=1

with open(os.path.join('./result/','reppoint_'+jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(eval_data,f)
