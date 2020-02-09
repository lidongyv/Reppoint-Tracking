import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import copy
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

def carocar():
	return 0
def caroperson():
	return 0

jsonfile_name='kitti_train_3class.json'
data_path='/backdata01/KITTI/kitti/tracking'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	train=json.load(f)
print(len(train))
jsonfile_name='kitti_val_3class.json'
data_path='/backdata01/KITTI/kitti/tracking'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	val=json.load(f)
print(len(val))
data=train+val
new_classes=['vehicle','person','cyclist']
new_classes=[1,2,3]
occlusion_ratio=np.linspace(0,1,10).tolist()
occlusion_count=[0 for i in range(10)]
for i in range(len(data)):
	box=data[i]['ann']['bboxes']
	label=data[i]['ann']['label']
	for j in new_classes:
		