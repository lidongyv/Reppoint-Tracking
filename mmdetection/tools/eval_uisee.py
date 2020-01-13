# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-25 19:24:06  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-25 19:24:06

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import cv2

data_path='/backdata01/mingzhu_rgb/'
file_name='truck.txt'
# test a video and show the results
with open(os.path.join(data_path,file_name),'r',encoding='utf-8') as f:
	data=f.read().splitlines()
config_file ='/home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_uisee_mt.py'
checkpoint_file='/home/ld/RepPoints/work_dirs/reppoints_moment_r101_dcn_fpn_kitti_mt_class3/epoch_29.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

compute_time=0
eval_data=[]
out_path='/home/ld/RepPoints/out/mingzhu_rgb_kitti'
if not os.path.exists(out_path):
	os.mkdir(os.path.join(out_path))
for i,(frame) in enumerate(data):
	print(i,'in',len(data))
	img = mmcv.imread(os.path.join(data_path,frame))
	result,box_loc = inference_detector(model, img)
	show_result(img, result, model.CLASSES, show=False,out_file=os.path.join(out_path,frame))
	# exit()
	continue
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
	

(
	ffmpeg
	.input(os.path.join(out_path,'*.png'), pattern_type='glob', framerate=15)
	.output(os.path.join(out_path,'video.mp4'))
	.run()
)
