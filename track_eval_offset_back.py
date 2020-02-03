# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-29 00:58:39  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-29 02:28:39

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
	if isinstance(bboxes1,list):
		bboxes1=np.array(bboxes1)
	if  isinstance(bboxes2,list):
		bboxes2=np.array(bboxes2)
	if bboxes1.shape[0]==0 or bboxes2.shape[0]==0:
		return np.zeros(np.max([bboxes1.shape[0],bboxes2.shape[0]]))
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
def trackiou(iou,max_id):
	idx=np.arange(iou.shape[0])
	idc=np.arange(iou.shape[1])
	id=np.argmax(iou,axis=1)
	value=np.max(iou,axis=1)
	id_check=np.argmax(iou,axis=0)
	if (id_check[id]==idx).all() and (id[id_check]==idc).all():
		id=np.where(value<0.2,-1,id)
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
				id=np.where(value<0.2,-1,id)
				return id
			else:
				value=np.max(iou,axis=1)
				count=np.bincount(id)
		id=np.where(value<0.2,-1,id)
		return id

def trackcd(cd,max_id):
	idx=np.arange(cd.shape[0])
	idc=np.arange(cd.shape[1])
	id=np.argmin(cd,axis=1)
	value=np.min(cd,axis=1)
	id_check=np.argmin(cd,axis=0)
	if (id_check[id]==idx).all() and (id[id_check]==idc).all():
		id=np.where(value>50,-1,id)
		return id
	else:
		count=np.bincount(id)
		while(np.max(count)>1):
			for i in range(np.min(id),np.max(id)+1):
				if count[i]<=1:
					continue
				ids=np.where(id==i)[0]
				cd[ids,i]=0
				cd[ids[np.argmin(value[ids])],i]=value[ids[np.argmin(value[ids])]]
				id=np.argmin(cd,axis=1)
			if (np.min(cd,axis=1)==value).all:
				count=np.bincount(id)
				for i in range(np.min(id),np.max(id)+1):
					if count[i]<=1:
						continue
					ids=np.where(id==i)[0]
					id[ids]=-1
					id[ids[np.argmin(value[ids])]]=i
				id=np.where(value>50,-1,id)
				return id
			else:
				value=np.min(cd,axis=1)
				count=np.bincount(id)
		id=np.where(value>50,-1,id)
		return id
def iou_cd(iou_id,cd_id,refer_id,max_id):
	# print(iou_id,cd_id,len(refer_id))
	iou_id=np.array(iou_id).astype(np.int)
	cd_id=np.array(cd_id).astype(np.int)
	refer_id=np.array(refer_id).astype(np.int)
	new_id=np.where(iou_id==-1)[0]
	#print(refer_id,iou_id)
	iou_id=refer_id[iou_id]
	iou_id[new_id]=-1
	new_id=np.where(cd_id==-1)[0]
	cd_id=refer_id[cd_id]
	cd_id[new_id]=-1
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
def target_in_box(point,bbox):
	check=[]
	#point xy,bbox yx
	for i in range(len(bbox)):
		if point[0]>bbox[i][1] and point[0]<bbox[i][3] and point[1]>bbox[i][0] and point[1]<bbox[i][2]:
			check.append(1)
		else:
			check.append(0)
	return np.array(check)
def pair_loc_det(loc,det):
	box=det[:,:4]
	ploc=loc[:,:2]
	center=(box[:,:2]+box[:,-2:])/2
	dist=np.sqrt(np.sum(np.square(ploc[:,None,:]-center[None,:,:]),axis=-1))
	index=np.argmin(dist,axis=1)
	return loc,det[index,:]
def track_init(refer,agg,support,record):
	# for m in range(10):
	# 	support.append([offset_data[m][i],
	# 	mask_data[m][i],
	# 	result_record[m+2][i],
	# 	loc_record[m+2][i]])
	# print(len(refer),len(agg),len(support))
	max_id=-1
	for i in range(-1,-11,-1):
		#from the frame 9 to frame 0
		box_r=refer[i]
		box_a=agg[i]
		offset_s=[]
		mask_s=[]
		box_s=[]
		loc_s=[]
		box_iou=[]
		record_s=[]
		
		if i==-1 or max_id==-1:
			print('init')
			print(i,max_id,box_r.shape)
			#init the first frame
			if box_r.shape[0]>0:
				#nms on the reference
				keep=nms(box_r)
				box_r=box_r[keep,:]
				boxm=[box_r,box_a]
				for j in range(10):
					#select support bbox
					# print(box_r.shape)
					# print(box_a.shape)
					# print(support[i][j][2].shape)
					boxm.append(support[i][j][2])
				#nms with support
				boxm=np.concatenate(boxm,axis=0)
				refer_len=box_r.shape[0]
				keep=TN_check(boxm,refer_len)
				box_r=box_r[keep,:]
				label=np.arange(box_r.shape[0])
				#add to record
				record[i]['ann']['bboxes']=box_r.tolist()
				record[i]['ann']['track_id']=label.tolist()
				max_id=np.max(label)
			else:
				label=box_r
				#add to record
				record[i]['ann']['bboxes']=box_r.tolist()
				record[i]['ann']['track_id']=label.tolist()
			print('trackid:',record[i]['ann']['track_id'])
		else:
			print('back search')
			#nms of refer
			box_r=refer[i]
			if box_r.shape[0]==0:
				label=box_r
				record[i]['ann']['bboxes']=box_r.tolist()
				record[i]['ann']['track_id']=label.tolist()
				continue
			keep=nms(box_r)
			box_r=box_r[keep,:]
			labels=np.arange(box_r.shape[0])*0-1
			box_a=agg[i]
			#select the support within 10 frames, i=2, -i-1=1, only one support frame
			for j in range(-i-1):
				offset_s.append(support[i][j][0])
				mask_s.append(support[i][j][1])
				box_s.append(support[i][j][2])
				loc_s.append(support[i][j][3])
				box_iou.append(bbox_overlaps(box_r,box_s[-1]))
				record_s.append(record[i+j+1])
			# print(box_s)

			for m in range(box_r.shape[0]):
				#align each bbox on refer det with a support label or a new label
				support_labels=[]
				support_score=[]
				for n in range(len(box_iou)):
					#check in each support detection result
					if np.max(box_iou[n][m])>=0.5:
						select=np.argmax(box_iou[n][m])
						print('max iou',select)
					else:
						print('new_iou')
						support_labels.append(-1)
						support_score.append(-1)
						continue
					#yx
					loc=loc_s[n][select]
					#xy
					offset=offset_s[n][select]
					mask=mask_s[n][select]
					# print(offset.shape,mask.shape)
					#18,9
					# print(loc)
					if np.max(mask)>=0.2:
						
						#select the offset with mask and compute the target bbox in support
						mask_select=np.where(mask>=0.2)[0]
						mask=mask[mask_select]
						print('max mask',mask)
						t=np.ones(2*mask.shape[0])
						for o in range(mask.shape[0]):
							t[2*o]=offset[2*mask_select[o]]
							t[2*o+1]=offset[2*mask_select[o]+1]
						offset=t
						target=[]
						#compute target point
						for o in range(mask.shape[0]):
							x=loc[1]+offset[2*o]*loc[2]
							y=loc[0]+offset[2*o+1]*loc[2]
							target.append([x,y])
						#select the support bbox by the target point
						support_select=np.zeros(box_s[n].shape[0])
						for o in range(mask.shape[0]):
							#each support box is align with a confidence by mask
							check=target_in_box(target[o],box_s[n])*mask[o]
							support_select+=check
						#select the bbox with max score and align the label
						if np.max(support_select)>=0.2:
							support_index=np.where(support_select>=0.2)
							# support_index=np.argmax(support_select)
							for o in range(len(support_index)):
								
								bbox_support=box_s[n][support_index[o]].reshape(1,5)
								iou_support=bbox_overlaps(bbox_support,record_s[n]['ann']['bboxes'])
								if np.max(iou_support)>0.5:
									support_index=np.argmax(iou_support)
								else:
									support_labels.append(-1)
									support_score.append(-1)
									continue


							support_labels.append(record_s[n]['ann']['track_id'][support_index])
							support_score.append( np.max(support_select))
						else:
							support_labels.append(-1)
							support_score.append(-1)
					else:
						mask=mask

						t=np.ones(2*mask.shape[0])
						x=0
						y=0
						for o in range(mask.shape[0]):
							x+=offset[2*o]*mask[o]
							y+=offset[2*o+1]*mask[o]
						target=[x+loc[1],y+loc[0]]
						#select the support bbox by the target point
						support_select=np.zeros(box_s[n].shape[0])
						#each support box is align with a confidence by mask
						check=target_in_box(target,box_s[n])
						support_select+=check
						#select the bbox with max score and align the label
						if np.max(support_select)==1:
							support_index=np.argmax(support_select)
							bbox_support=box_s[n][support_index].reshape(1,5)
							iou_support=bbox_overlaps(bbox_support,record_s[n]['ann']['bboxes'])
							if np.max(iou_support)>0.5:
								support_index=np.argmax(iou_support)
							else:
								support_labels.append(-1)
								support_score.append(-1)
								continue
							support_labels.append(record_s[n]['ann']['track_id'][support_index])
							support_score.append(1/9)
						else:
							support_labels.append(-1)
							support_score.append(-1)
				#select lable with max score
				if np.max(support_labels)>=0:
					select_score=np.argmax(support_score)
					labels[m]=support_labels[select_score]
				else:
					labels[m]=-1

			new_object=np.where(labels==-1)[0]
			for m in range(len(new_object)):
				max_id+=1
				labels[new_object[m]]=max_id
			
			record[i]['ann']['bboxes']=box_r.tolist()
			record[i]['ann']['track_id']=labels.tolist()
			print('trackid:',record[i]['ann']['track_id'])
	return record,max_id
def track_online(refer,agg,support,record,info,max_id):
	offset_s=[]
	mask_s=[]
	box_s=[]
	loc_s=[]
	box_iou=[]
	if refer.shape[0]==0:
		info['ann']['bboxes']=refer.tolist()
		info['ann']['track_id']=refer.tolist()
		return info,max_id
	else:
		box_r=refer
		keep=nms(box_r)
		box_r=box_r[keep,:]
		labels=np.arange(box_r.shape[0])*0-1
		box_a=agg
		#select the support within 10 frames, i=2, -i-1=1, only one support frame
		for i in range(10):
			offset_s.append(support[i][0])
			mask_s.append(support[i][1])
			box_s.append(support[i][2])
			loc_s.append(support[i][3])
			box_iou.append(bbox_overlaps(box_r,box_s[-1]))
			record_s=record

		for m in range(box_r.shape[0]):
			#align each bbox on refer det with a support label or a new label
			support_labels=[]
			support_score=[]
			for n in range(10):
				#check in each support detection result
				if np.max(box_iou[n][m])>=0.5:
					select=np.argmax(box_iou[n][m])
				else:
					support_labels.append(-1)
					support_score.append(-1)
					continue
				#yx
				loc=loc_s[n][select]
				#xy
				offset=offset_s[n][select]
				mask=mask_s[n][select]
				# print(offset.shape,mask.shape)
				#18,9
				# print(loc)
				if np.max(mask)>=0.2:
					#select the offset with mask and compute the target bbox in support
					mask_select=np.where(mask>=0.2)[0]
					mask=mask[mask_select]
					t=np.ones(2*mask.shape[0])
					for o in range(mask.shape[0]):
						t[2*o]=offset[2*mask_select[o]]
						t[2*o+1]=offset[2*mask_select[o]+1]
					offset=t
					target=[]
					#compute target point
					for o in range(mask.shape[0]):
						x=loc[1]+offset[2*o]*loc[2]
						y=loc[0]+offset[2*o+1]*loc[2]
						target.append([x,y])
					#select the support bbox by the target point
					support_select=np.zeros(box_s[n].shape[0])
					for o in range(mask.shape[0]):
						#each support box is align with a confidence by mask
						check=target_in_box(target[o],box_s[n])*mask[o]
						support_select+=check
					#select the bbox with max score and align the label
					if np.max(support_select)>=0.2:
						support_index=np.argmax(support_select)
						# print(record_s[n]['ann']['track_id'])
						# print(support_index)
						bbox_support=box_s[n][support_index].reshape(1,5)
						iou_support=bbox_overlaps(bbox_support,record_s[n]['ann']['bboxes'])
						if np.max(iou_support)>0.5:
							support_index=np.argmax(iou_support)
						else:
							support_labels.append(-1)
							support_score.append(-1)
							continue
						support_labels.append(record_s[n]['ann']['track_id'][support_index])
						support_score.append(np.max(support_select))
					else:
						support_labels.append(-1)
						support_score.append(-1)
				else:
					x=0
					y=0
					for o in range(mask.shape[0]):
						x+=offset[2*o]*mask[o]
						y+=offset[2*o+1]*mask[o]
					target=[x+loc[1],y+loc[0]]
					#select the support bbox by the target point

					#each support box is align with a confidence by mask
					support_select=target_in_box(target,box_s[n])

					#select the bbox with max score and align the label
					if np.max(support_select)==1:
						support_index=np.argmax(support_select)
						# print(record_s[n]['ann']['track_id'])
						# print(support_index)
						bbox_support=box_s[n][support_index].reshape(1,5)
						iou_support=bbox_overlaps(bbox_support,record_s[n]['ann']['bboxes'])
						if np.max(iou_support)>0.5:
							support_index=np.argmax(iou_support)
						else:
							support_labels.append(-1)
							support_score.append(-1)
							continue
						support_labels.append(record_s[n]['ann']['track_id'][support_index])
						support_score.append(1)
					else:
						support_labels.append(-1)
						support_score.append(-1)
			#select lable with max score
			if np.max(support_labels)>=0:
				select_score=np.argmax(support_score)
				labels[m]=support_labels[select_score]
			else:
				labels[m]=-1
		new_object=np.where(labels==-1)[0]
		for m in range(len(new_object)):
			max_id+=1
			labels[new_object[m]]=max_id
		
		info['ann']['bboxes']=box_r.tolist()
		info['ann']['track_id']=labels.tolist()
		print('trackid:',info['ann']['track_id'])
		return info,max_id
out_path='/home/ld/RepPoints/final/epoch13 thres0.2'
out_name=['refer','agg']
for i in range(10):
	out_name.append('frame_'+str(i+1))

result_record=[]
for i in range(12):
	result_record.append([])
for i in range(12):
	result_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'det_result.pkl'))
	# for m in range(len(result_record[i])):
	# 	for n in range(len(result_record[i][m])):
	# 		result_record[i][m][n]=result_record[i][m][n].numpy()
	print('det',len(result_record[i]))
offset_data=[]
for i in range(10):
	offset_data.append([])
for i in range(10):
	offset_data[i]=mmcv.load(os.path.join(out_path,out_name[i+2],'offset.pkl'))
	for m in range(len(offset_data[i])):
		for n in range(len(offset_data[i][m])):
			offset_data[i][m][n]=offset_data[i][m][n].numpy()
	print('offset',len(offset_data[i]))
mask_data=[]
for i in range(10):
	mask_data.append([])
for i in range(10):
	mask_data[i]=mmcv.load(os.path.join(out_path,out_name[i+2],'mask.pkl'))
	for m in range(len(mask_data[i])):
		for n in range(len(mask_data[i][m])):
			mask_data[i][m][n]=mask_data[i][m][n].numpy()
	print('mask',len(mask_data[i]))
loc_record=[]
for i in range(12):
	loc_record.append([])
for i in range(12):
	loc_record[i]=mmcv.load(os.path.join(out_path,out_name[i],'loc_result.pkl'))
	for m in range(len(loc_record[i])):
		loc_record[i][m]=loc_record[i][m].data.cpu().numpy()
	print('loc',len(loc_record[i]))
video_name=None
video_length=0
json_data_iou=[]
data=mmcv.load(os.path.join(out_path,out_name[0],'track.pkl'))
print('track',len(data))
track_record=[]
init_support=[]
init_refer=[]
init_record=[]
init_agg=[]
for i in range(len(data)):
	print(i,'in',len(data))
	info=data[i]
	
	if info['filename'].split('/')[2]!=video_name:
		#init from the first ten frames
		init_support=[]
		init_refer=[]
		init_record=[]
		init_agg=[]
		video_length=1
		video_name=info['filename'].split('/')[2]
		max_id=-1
		support=[]
		for m in range(10):
			det_box=np.vstack(result_record[m+2][i])
			det_loc=loc_record[m+2][i]
			if det_box.shape[0]>0:
				det_loc,det_box=pair_loc_det(det_loc,det_box)
			support.append([offset_data[m][i],
			mask_data[m][i],
			det_box,
			det_loc])
		# print(result_record[0][i])
		init_support.append(support)
		init_refer.append(np.vstack(result_record[0][i]))
		init_agg.append(np.vstack(result_record[1][i]))
		init_record.append(info)
		print('video_name',video_name,'image_name',info['filename'],'processing frame',video_length)
	else:
		video_length+=1
		print('video_name',video_name,'image_name',info['filename'],'processing frame',video_length)
		if video_length<11:
			support=[]
			for m in range(10):
				support.append([offset_data[m][i],
				mask_data[m][i],
				np.vstack(result_record[m+2][i]),
				loc_record[m+2][i]])
			init_support.append(support)
			init_refer.append(np.vstack(result_record[0][i]))
			init_agg.append(np.vstack(result_record[1][i]))
			init_record.append(info)

		if video_length==10:
			init_record,max_id=track_init(init_refer,init_agg,init_support,init_record)
			track_record=track_record+init_record
			# print(track_record)
			init_support=[]
			init_refer=[]
			init_record=[]
			init_agg=[]
			continue

		if video_length>10:
			# i=i+1000
			# print(data[i]['filename'])
			track_refer=np.vstack(result_record[0][i])
			track_agg=np.vstack(result_record[1][i])
			support=[]
			for m in range(10):
				det_box=np.vstack(result_record[m+2][i])
				det_loc=loc_record[m+2][i]
				if det_box.shape[0]>0:
					det_loc,det_box=pair_loc_det(det_loc,det_box)
				support.append([offset_data[m][i],
				mask_data[m][i],
				det_box,
				det_loc])
			info,max_id=track_online(track_refer,track_agg,support,track_record[-10:][::-1],info,max_id)
			track_record.append(info)
			

		if video_length>30:
			with open(os.path.join(out_path,'offset_tracking_result.json'),'w+',encoding='utf-8') as f:
				data=json.dump(track_record,f)
			exit()


with open(os.path.join(out_path,'offset_tracking_result.json'),'w+',encoding='utf-8') as f:
	data=json.dump(track_record,f)

