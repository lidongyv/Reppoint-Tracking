import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
jsonfile_name='kitti_train.json'
data_path='/backdata01/KITTI/kitti/tracking'
# test a video and show the results
with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
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
# test:    [8614, 1307,    145,         8897,        509,           681,      417, 532]
new_classes=['vehicle','person','cyclist']
# train [20680, 2740, 1257]
# test  [9921, 9406, 681]
count=[0,0,0,0]
train_count=[0,0,0,0,0,0,0,0]
for i in range(len(data)):
	if len(data[i]['ann']['labels'])==0:
		continue
	labels=data[i]['ann']['labels']
	for j in range(len(labels)):
		train_count[labels[j]-1]+=1
		if labels[j]==1 or labels[j]==2:
			labels[j]=1
			count[0]+=1
		elif labels[j]==4 or labels[j]==5:
			labels[j]=2
			count[1]+=1
		elif labels[j]==6:
			labels[j]=3
			count[2]+=1
		else:
			labels[j]=0
			count[3]+=1
	data[i]['ann']['labels']=labels
print(count)
print(train_count)
jsonfile_name='kitti_train_3class.json'
with open(os.path.join(data_path,jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(data,f)