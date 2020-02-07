 # -*- coding: utf-8 -*-  
 #@Author: lidong yu  
 #@Date:':2020-02-05':22:12:45 
 #@Last Modified by: lidong yu 
 #@Last Modified time:':2020-02-05':22:12:45

import numpy as np
import os
import cv2
import scipy.io
import  xml.dom.minidom
import json
# mat = scipy.io.loadmat('/backdata01/ILSVRC2015/ILSVRC2015_devkit/ILSVRC2015/devkit/data/ILSVRC2015_vid_validation_ground_truth.mat')
# print(mat)
name_map={
'n02691156':1 ,
'n02419796':2 ,
'n02131653':3 ,
'n02834778': 4 ,
'n01503061': 5 ,
'n02924116': 6 ,
'n02958343': 7 ,
'n02402425': 8 ,
'n02084071': 9 ,
'n02121808':10 ,
'n02503517':11 ,
'n02118333':12 ,
'n02510455':13 ,
'n02342885':14 ,
'n02374451':15 ,
'n02129165':16 ,
'n01674464':17 ,
'n02484322':18 ,
'n03790512':19 ,
'n02324045':20 ,
'n02509815':21 ,
'n02411705':22 ,
'n01726692':23 ,
'n02355227':24 ,
'n02129604':25 ,
'n04468005':26 ,
'n01662784':27 ,
'n04530566':28 ,
'n02062744':29 ,
'n02391049':30 }
vid_path='/backdata01/ILSVRC2015'
train_data='/backdata01/ILSVRC2015/ImageSets/VID/train'
val_data='/backdata01/ILSVRC2015/ImageSets/VID/val.txt'
train_files=os.listdir(train_data)
train_data_all=[]
for i in range(len(train_files)):
	with open(os.path.join(train_data,train_files[i])) as f:
		data= f.read().splitlines()
		for j in range(len(data)):
			data[j]=data[j].split(' ')[0]
	train_data_all+=data
train_data_all.sort()
print(len(train_data_all))
all_data=train_data_all
train_data_all=[]
for i in range(len(all_data)):
	if all_data[i] not in train_data_all:
		train_data_all.append(all_data[i])
print(len(train_data_all))
# print(train_data_all)
train_img_path='/backdata01/ILSVRC2015/Data/VID/train'
train_ann_path='/backdata01/ILSVRC2015/Annotations/VID/train'
train_img_all=[[] for i in range(len(train_data_all))]
train_ann_all=[[] for i in range(len(train_data_all))]
count=0
for i in range(len(train_data_all)):
	train_img_all[i]=[os.path.join(train_data_all[i],m) for m in os.listdir(os.path.join(train_img_path,train_data_all[i]))]
	train_img_all[i].sort()
	train_ann_all[i]=[m.split('.')[0]+'.xml' for m in train_img_all[i]]
	count+=len(train_img_all[i])
# print(count)
# print(train_img_all[0][0],train_ann_all[0][0])
data=[]
for m in range(len(train_ann_all)):
	for n in range(len(train_ann_all[m])):
		dom = xml.dom.minidom.parse(os.path.join(train_ann_path,train_ann_all[m][n]))
		root = dom.documentElement
		size = root.getElementsByTagName('size')[0]
		width=int(size.getElementsByTagName('width')[0].childNodes[0].data)
		height=int(size.getElementsByTagName('height')[0].childNodes[0].data)
		video_id=root.getElementsByTagName('folder')[0].childNodes[0].data
		frame_id=root.getElementsByTagName('filename')[0].childNodes[0].data
		objects = root.getElementsByTagName('object')
		tmp_bbox=[]
		tmp_label=[]
		tmp_track=[]
		for i in range(len(objects)):
			obj=objects[i]
			id=int(obj.getElementsByTagName('trackid')[0].childNodes[0].data)
			name=name_map[obj.getElementsByTagName('name')[0].childNodes[0].data]
			bbox=obj.getElementsByTagName('bndbox')[0]
			xmin=int(bbox.getElementsByTagName('xmin')[0].childNodes[0].data)
			xmax=int(bbox.getElementsByTagName('xmax')[0].childNodes[0].data)
			ymin=int(bbox.getElementsByTagName('ymin')[0].childNodes[0].data)
			ymax=int(bbox.getElementsByTagName('ymax')[0].childNodes[0].data)
			tmp_bbox.append([xmin,ymin,xmax,ymax])
			tmp_label.append(name)
			tmp_track.append(id)
		print(train_img_all[m][n])
		data.append({
			'filename':train_img_all[m][n],
			'width':width,
			'height':height,
			'video_id':video_id,
			'frame_id':frame_id,
			'ann':{
				'bboxes':tmp_bbox,
				'labels':tmp_label,
				'track_id':tmp_track,
				'track_id_ignore':[],
				'bboxes_ignore':[],
				
				}
		})
jsonfile_name='vid_train_all_class.json'
with open(os.path.join(vid_path,jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(data,f)

# with open(os.path.join(train_data,val_data)) as f:
# 	val_data_all = f.read().splitlines()
# 	for j in range(len(val_data_all)):
# 		val_data_all[j]=val_data_all[j].split(' ')[0]
# print(len(val_data_all))
