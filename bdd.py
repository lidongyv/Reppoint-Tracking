import os
import numpy as np
import json
import shutil
import cv2
cv2.imread()
exit()
# bdd class
# "bike",
# "bus",
# "car",
# "motor",
# "person",
# "rider",
# "traffic light",
# "traffic sign",
# "train",
# "truck"

vehicle=['bus','car','train','truck']
pedestrain=['person','rider']
cyclist=['bike','motor']
path='/backdata01/bdd/bdd100k/tracking_cvpr2019/'
label=path+'bdd100k_tracking_cvpr2019_val.json'
with open(label) as f:
	label=json.load(f)
print(label[0])
data=[]
video_check='1'
for i in range(len(label)):
	data_t=label[i]
	if video_check!=data_t['videoName']:
		os.mkdir(path+'video/'+data_t['videoName'])
		video_check=data_t['videoName']
	old_path=path+'image/'+data_t['name']
	new_path=path+'video/'+data_t['videoName']+data_t['name']
	shutil.copy2(old_path,new_path)
	data.append({
		'video_id':data_t['videoName'],
		'file_name':data_t['name'],

	})
path='/backdata01/KITTI/kitti/tracking/'
label=path+'kitti_train_3class.json'
with open(label) as f:
	label=json.load(f)
print(label[0])

# label='/home/ld/RepPoints/data/coco/annotations/instances_val2017.json'
# with open(label) as f:
# 	label=json.load(f)['annotations']
# print(label[0])