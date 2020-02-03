import numpy as np
import torch
import os
import time
import json
import cv2
from IO import *
import mmcv
import matplotlib.pyplot as plt

def nms(dets, thresh=0.5):
	x1 = dets[:, 1]
	y1 = dets[:, 0]
	x2 = dets[:, 3]
	y2 = dets[:, 2]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)

		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]

	return keep
def check(dets, refer_len,thresh=0.5):
	x1 = dets[:, 1]
	y1 = dets[:, 0]
	x2 = dets[:, 3]
	y2 = dets[:, 2]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= thresh)[0]
		check=len(np.where(ovr >= thresh)[0])
		# print('thres',check)
		# if i<refer_len:
		# 	check_len=1
		# else:
		# 	check_len=2
		# check=len(np.where(np.where(ovr >= thresh)[0]>refer_len)[0])
		# print('length',check)
		if check>1:
			keep.append(i)
		elif scores[i]>0.9:
			keep.append(i)

		# keep.append(i)
		order = order[inds + 1]
	# print(dets.shape,len(keep))
	return keep

def FP_check(dets,thresh=0.5):
	x1 = dets[:, 1]
	y1 = dets[:, 0]
	x2 = dets[:, 3]
	y2 = dets[:, 2]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= thresh)[0]
		check=len(np.where(ovr >= thresh)[0])

		if check>1:
			keep.append(i)
		# elif scores[i]>0.9:
		# 	keep.append(i)
		order = order[inds + 1]
	return keep
def TN_check(dets,refer_len, thresh=0.5):
	x1 = dets[:, 1]
	y1 = dets[:, 0]
	x2 = dets[:, 3]
	y2 = dets[:, 2]
	scores = dets[:, 4]

	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores[:refer_len].argsort()[::-1]
	order_all=np.array(order.tolist()+np.arange(refer_len,dets.shape[0]).tolist())
	# print(order)
	keep = []
	while order.size > 0:
		i = order[0]
		xx1 = np.maximum(x1[i], x1[order_all[1:]])
		yy1 = np.maximum(y1[i], y1[order_all[1:]])
		xx2 = np.minimum(x2[i], x2[order_all[1:]])
		yy2 = np.minimum(y2[i], y2[order_all[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		ovr = inter / (areas[i] + areas[order_all[1:]] - inter)
		inds = np.where(ovr <= thresh)[0]
		# print(inds)
		check=len(np.where(ovr >= thresh)[0])
		# print('thres',check)
		# if i<refer_len:
		# 	check_len=1
		# else:
		# 	check_len=2
		# check=len(np.where(np.where(ovr >= thresh)[0]>refer_len)[0])
		# print('length',check)
		if check>1:
			keep.append(i)
		# elif scores[i]>0.5:
		# 	keep.append(i)
		order = order[1:]
		order_all = order_all[inds + 1]
	# print(dets.shape,len(keep))
	return keep
data_path='/backdata01/KITTI/kitti/tracking'
jsonfile_name='kitti_val_3class.json'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
out_name=['refer','agg']
for i in range(10):
	out_name.append('frame_'+str(i+1))
out_path='/home/ld/RepPoints/final/epoch13 thres0.1'
result_record=[]
for i in range(12):
	result_record.append([])
for i in range(12):
	result_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'det_result.pkl'))
video_name=None
video_length=0
max=0
fuse=result_record[0]
track=mmcv.load(os.path.join(out_path,out_name[0],'track.pkl'))
for i in range(len(data)):
	#left-yx,right-yx,score, three list for three classes
	print(i)
	refer=result_record[0][i]
	agg=result_record[1][i]
	support_result=[]
	# stack=[]
	# for m in range(3):
	# 	for n in range(len(refer[m])):
	# 		stack.append(np.array(refer[m][n].tolist()+[m]))
	# refer=np.stack(stack)
	# print(refer.shape)
	for j in range(10):
		support_result.append(result_record[j+2][i])
	for m in range(3):
		refer_len=refer[m].shape[0]
		boxm=[refer[m][nms(refer[m])],agg[m][nms(agg[m])]]
		for j in range(10):
			# print(support_result[j][m].shape)
			boxm.append(support_result[j][m][nms(support_result[j][m])])
		boxm=np.concatenate(boxm,axis=0)
		# fuse[i][m]=boxm[nms(boxm)]
		keep_tn=TN_check(boxm,refer_len)
		keep_fp=FP_check(boxm[refer_len:])
		boxtn=boxm[keep_tn,:]
		fuse[i][m]=boxtn
		boxfp=boxm[refer_len:,:][keep_fp,:]
		fuse[i][m]=boxfp
		box_fuse=np.concatenate([boxtn,boxfp],axis=0)
		keep=nms(box_fuse)
		fuse[i][m]=box_fuse[keep,:]
	
	bboxes = np.vstack(fuse[i])
	scores = bboxes[:, -1]
	inds = scores >= 0
	scores=bboxes[inds, :][:,4:]
	bboxes = bboxes[inds, :][:,:4]
	labels = [
		np.full(bbox.shape[0], i, dtype=np.int32)
		for i, bbox in enumerate(fuse[i])
	]
	labels = np.concatenate(labels)
	labels = labels[inds]
	track[i]['ann']['bboxes']=bboxes.tolist()
	track[i]['ann']['labels']=labels.tolist()
	# print(labels.tolist())
	# exit()
	track[i]['ann']['track_id']=labels.tolist()
	track[i]['ann']['score']=scores.tolist()

mmcv.dump(fuse, os.path.join(out_path,'fuse_result.pkl'))
mmcv.dump(track, os.path.join(out_path,'track_result.pkl'))
	# print(len(refer))
	# print(len(agg))
	# print(refer)
	# exit()
