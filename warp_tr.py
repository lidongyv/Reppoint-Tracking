# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-12-10 21:14:09  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-12-10 21:14:09

import numpy as np
import torch
import os
import time
import json
import cv2
from IO import *
import mmcv
import matplotlib.pyplot as plt
def bbox_overlaps(bboxes1, bboxes2):
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
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
def box2center(bbox):

	# center=[]
	# for i in bbox:
	# 	left_top = np.array(i[0], i[1])
	# 	right_bottom = np.array(i[2], i[3])
	# 	center.append((left_top+right_bottom)/2)
	# return np.array(center)
	return (bbox[:,:2]+bbox[:,-2:])/2
def get_transform(flow,bbox):
	transform=flow[np.floor(c0[:,1]).astype(np.int),np.floor(c0[:,0]).astype(np.int)]
	return transform
def center_dist(c0,c1):
	dist=np.sqrt(np.sum(np.square(c0[:,None,:]-c1[None,:,:]),axis=-1))
	return dist
def bbox_select(flow0,bbox):
	#flow0 is the origin flow, we check the warp flow and compute the largest overlap
	#with the detection bbox
	return 0

	idx=np.arange(iou.shape[0])
	idc=np.arange(iou.shape[1])
	id=np.argmax(iou,axis=1)
	value=np.max(iou,axis=1)
	id_check=np.argmax(iou,axis=0)
	if (id_check[id]==idx).all() and (id[id_check]==idc).all():
		id=np.where(value<0.1,-1,id)
		return id
	else:
		count=np.bincount(id)
		while(np.max(count)>1):
			for i in range(np.min(id),np.max(id)+1):
				if count[i]<=1:
					continue
				ids=np.where(id==i)[0]
				iou[ids,i]=0
				iou[ids[np.argmax(value[ids])],i]=value[ids[np.argmax(value[ids])]]
			id=np.argmax(iou,axis=1)
			if (np.max(iou,axis=1)==value).all():
				count=np.bincount(id)
				for i in range(np.min(id),np.max(id)+1):
					if count[i]<=1:
						continue
					ids=np.where(id==i)[0]
					id[ids]=-1
					id[ids[np.argmax(value[ids])]]=i
				id=np.where(value<0.1,-1,id)
				return id
			else:
				value=np.max(iou,axis=1)
				count=np.bincount(id)
		id=np.where(value<0.3,-1,id)
		return id


	# print(iou_id,cd_id,len(refer_id))
	iou_id=np.array(iou_id).astype(np.int)

	refer_id=np.array(refer_id).astype(np.int)
	new_id=np.where(iou_id==-1)[0]
	#print(refer_id,iou_id)
	iou_id=refer_id[iou_id]
	iou_id[new_id]=-1

	if (iou_id==cd_id).all():
		id=iou_id
		new_id=np.array(new_id)
		for i in range(new_id.shape[0]):
			max_id+=1
			iou_id[new_id[i]]=max_id
		return id
	else:
		id=np.where(iou_id==cd_id,iou_id,-1)
		check=np.where(id==-1)[0]
		for i in check:
			if iou_id[i]==-1 or cd_id[i]==-1:
				max_id+=1
				id[i]=max_id
			else:
				if iou_id[i]==-1:
					max_id+=1
					id[i]=max_id
				else:
					id[i]=iou_id[i]

		return id
def trackiou(iou,max_id):
	idx=np.arange(iou.shape[0])
	idc=np.arange(iou.shape[1])
	id=np.argmax(iou,axis=1)
	value=np.max(iou,axis=1)
	id_check=np.argmax(iou,axis=0)
	if (id_check[id]==idx).all() and (id[id_check]==idc).all():
		id=np.where(value<0.1,-1,id)
		return id
	else:
		count=np.bincount(id)
		while(np.max(count)>1):
			for i in range(np.min(id),np.max(id)+1):
				if count[i]<=1:
					continue
				ids=np.where(id==i)[0]
				iou[ids,i]=0
				iou[ids[np.argmax(value[ids])],i]=value[ids[np.argmax(value[ids])]]
			id=np.argmax(iou,axis=1)
			if (np.max(iou,axis=1)==value).all():
				count=np.bincount(id)
				for i in range(np.min(id),np.max(id)+1):
					if count[i]<=1:
						continue
					ids=np.where(id==i)[0]
					id[ids]=-1
					id[ids[np.argmax(value[ids])]]=i
				id=np.where(value<0.1,-1,id)
				return id
			else:
				value=np.max(iou,axis=1)
				count=np.bincount(id)
		id=np.where(value<0.3,-1,id)
		return id
def iouboxid(iou_id,refer_id,max_id):
	iou_id=np.array(iou_id).astype(np.int)
	refer_id=np.array(refer_id).astype(np.int)
	new_id=np.where(iou_id==-1)[0]
	#print(refer_id,iou_id)
	iou_id=refer_id[iou_id]
	iou_id[new_id]=-1
	id=iou_id
	new_id=np.array(new_id)
	for i in range(new_id.shape[0]):
		max_id+=1
		iou_id[new_id[i]]=max_id
	return id
def tracktr(iou,tr,box1,box0):
	#iou is box1 to box0
	#tr is box0 and box1
	id=[]
	tr0=tr[0,:,:]
	tr1=tr[1,:,:]
	for i in range(box1.shape[0]):
		if np.sum(iou[i])==0:
			id.append(-1)
			break
		j=np.argmax(iou[i])
		value=np.max(iou[i])
		while(value>0):
			iou[i,j]=0
		
	return 0
def crop(tr,box):
	return tr[box[0]:box[2],box[1]:box[3]]
data_path='/backdata01/KITTI/kitti/tracking'
# ['calib', 'disparity', 'image_02', 'label_02', 'splits', 'velodyne', 'velodyne_img', 
# 'Flow', 'Inv_Flow', 'trackinglabel_02', 'kitti.json', 'shuffle data', 'kitti_train.json'
#  'kitti_val.json']

jsonfile_name='kitti_val.json'
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
video_name=None
video_length=0
json_data_iou=[]
json_data_flow=[]
for i in range(len(data)):
	print(i,'in',len(data))
	info=data[i]
	if info['video_id']!=video_name:
		video_length=1
		video_name=info['video_id']
		max_id=-1
		max_id_iou=-1
		max_id_flow=-1
		trajectory=torch.load('/backdata01/KITTI/kitti/trajectory/'+video_name+'.pkl')
		trajectory=trajectory['trajectory volume'].data.cpu().numpy()
	else:
		video_length+=1

	print('video_id',video_name,'image_name',info['filename'],'processing frame',video_length)
	print('dection result:',np.array(info['ann']['track_id']))
	if np.array(info['ann']['bboxes']).shape[0]==0:
		result={"video_id":info['video_id'],"filename":os.path.join(info['filename']), \
			"ann":{"bboxes":[],"labels":[], \
				"track_id":[]}}
		json_data_iou.append(result)

		print('no detection')
		continue
	elif np.array(data[i-1]['ann']['bboxes']).shape[0]==0:
		boxid=[]
		for j in range(np.array(info['ann']['bboxes']).shape[0]):
			boxid.append(j)
		boxid=np.array(boxid)
		result={"video_id":info['video_id'],"filename":os.path.join(info['filename']), \
			"ann":{"bboxes":info['ann']['bboxes'],"labels":[], \
				"track_id":(boxid+max_id_iou+1).tolist()}}
		json_data_iou.append(result)

		print('first detection')
		continue

	if video_length==1 or max_id==-1:
		print('first frame')
		boxid=[]
		boxt0=np.array(info['ann']['bboxes'])
		for j in range(boxt0.shape[0]):
			boxid.append(j)
		if len(boxid)>0:
			max_id_iou=np.max(boxid)
			max_id=1
		else:
			max_id_iou=-1
			max_id_flow=-1
		result={"video_id":info['video_id'],"filename":os.path.join(info['filename']), \
			"ann":{"bboxes":boxt0.tolist(),"labels":info['ann']['labels'], \
				"track_id":boxid}}
		json_data_iou.append(result)
	else:
		refer_id=json_data_iou[-1]['ann']['track_id']
		boxt1=np.array(info['ann']['bboxes'])
		boxt0=np.array(data[i-1]['ann']['bboxes'])
		iou=bbox_overlaps(boxt1,boxt0)
		iou_id=tracktr(iou,trajectory[video_length-2:video_length],boxt1,boxt0)
		boxid=iouboxid(iou_id,refer_id,max_id)
		max_id_iou=np.max(np.maximum(boxid,max_id_iou))
		print('iou id:',boxid,boxid.shape)
		result={"video_id":info['video_id'],"filename":os.path.join(info['filename']), \
			"ann":{"bboxes":boxt1.tolist(),"labels":info['ann']['labels'], \
				"track_id":boxid.tolist()}}
		json_data_iou.append(result)

with open(os.path.join('./result/','reppoint_tr_'+jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(json_data_iou,f)

