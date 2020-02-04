import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import copy
jsonfile_name='kitti_train.json'
data_path='/backdata01/KITTI/kitti/tracking'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	train=json.load(f)
print(len(train))
jsonfile_name='kitti_val.json'
data_path='/backdata01/KITTI/kitti/tracking'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	val=json.load(f)
print(len(val))
data=train+val
max=0
min=100
# for i in range(len(data)):
# 	if len(data[i]['ann']['labels'])>0:
# 		if np.max(data[i]['ann']['labels'])>max:
# 			max=np.max(data[i]['ann']['labels'])
# 		if np.min(data[i]['ann']['labels'])<min:
# 			min=np.min(data[i]['ann']['labels'])
# print(max,min)
#8,1
# CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist', 'Tram', 'Misc', 'Person',)
classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
#train:   [18686, 1994, 1044, 2573, 167, 1257, 178, 261]
# test:	[8614, 1307,	145,		 8897,		509,		   681,	  417, 532]
new_classes=['vehicle','person','cyclist']
# train [20680, 2740, 1257]
# test  [9921, 9406, 681]
count=[]
video_id=[]
class_count=[]
for i in range(3):
	class_count.append(0)
for i in range(21):
	count.append(class_count.copy())
	video_id.append([])
video_count=-1
for i in range(len(data)):
	if len(data[i]['ann']['labels'])==0:
		continue
	labels=data[i]['ann']['labels']
	video=data[i]['video_id']
	if video_id[video_count]!=video:
		video_count+=1
		video_id[video_count]=video

	for j in range(len(labels)):
		if labels[j]==1 or labels[j]==2:
			labels[j]=1
		elif labels[j]==4 or labels[j]==5:
			labels[j]=2
		elif labels[j]==6:
			labels[j]=3
		else:
			labels[j]=0
		if labels[j]>0:
   			count[video_count][labels[j]-1]+=1
	data[i]['ann']['labels']=labels
# for i in range(21):
# 	print(video_id[i])
# 	print(new_classes)
# 	print(count[i])
test_name=['0001','0002','0009','0013','0016']
test_id=[1,2,9,13,16]
train=[]
test=[]
print(len(data))
for i in range(len(data)):
	video=data[i]['video_id']
	if video in test_name:
		test.append(data[i])
	else:
		train.append(data[i])
print(len(test))
test_count=[]
for i in range(21):
	if i in test_id:
		test_count.append(count[i])
print(np.sum(np.array(test_count),axis=0))
print(len(train))
train_count=[]
for i in range(21):
	if i not in test_id:
		train_count.append(count[i])
print(np.sum(np.array(train_count),axis=0))
jsonfile_name='kitti_val_3class.json'
with open(os.path.join(data_path,jsonfile_name),'w+',encoding='utf-8') as f:
	test=json.dump(test,f)
jsonfile_name='kitti_train_3class.json'
with open(os.path.join(data_path,jsonfile_name),'w+',encoding='utf-8') as f:
	train=json.dump(train,f)