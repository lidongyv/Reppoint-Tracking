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
count=0
for i in range(len(data)):
	# print(data[i])
	# exit()
	if len(data[i]['ann']['labels'])==0:
		continue
	labels=data[i]['ann']['labels']
	for j in range(len(labels)):
		if labels[j]==1 or labels[j]==2 or labels[j]==3 or labels[j]==6:
			labels[j]=1
		if labels[j]==4 or labels[j]==8:
			labels[j]=2
		if labels[j]==5:
			labels[j]=3
		if labels[j]==7:
			count+=1
	data[i]['ann']['labels']=labels
print(count)
jsonfile_name='kitti_train_3class.json'
with open(os.path.join(data_path,jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(data,f)