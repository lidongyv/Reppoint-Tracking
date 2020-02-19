import json
import mmcv
import numpy as np
# data=mmcv.load('/backdata01/kitti_bdd_waymo_2class.json')
data=mmcv.load('/backdata01/kitti_bdd_waymo_2class_val.json')
frame_count=[0 for i in range(3)]
class_count=[[0,0] for i in range(3)]
for i in range(len(data)):
	name=data[i]['filename'].split('/')
	if 'kitti' in name:
		dataset=0
	if 'bdd' in name:
		dataset=1
	if 'waymo' in name:
		dataset=2
	frame_count[dataset]+=1
	label=data[i]['ann']['labels']
	for j in range(len(label)):
		if label[j]==1:
			class_count[dataset][0]+=1
		if label[j]==2:
			class_count[dataset][1]+=1
print('all frames',np.sum(frame_count),'all bboxes','vehicle:',class_count[0][0]+class_count[1][0]+class_count[2][0], \
	'person:',class_count[0][1]+class_count[1][1]+class_count[2][1])
print('kitti frame:',frame_count[0],'kitti_class',class_count[0])
print('bdd frame:',frame_count[1],'bdd',class_count[1])
print('waymo frame:',frame_count[2],'waymo',class_count[2])
# kitti frame: 4251 kitti_class [12068, 4014]
# bdd frame: 8081 bdd [51997, 3118]
# waymo frame: 53360 waymo [578300, 339226]