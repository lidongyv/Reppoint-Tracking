import os
import numpy as np
import json
import shutil
import cv2
import mmcv
classes=['other','vehicle','person','cyclist']
label=[i for i in range(len(classes))]
class_count=[0 for i in range(len(classes))]
class_count=dict(zip(classes,class_count))
path='/backdata01/bdd/'
jsonfile_name='bdd_all_3class.json'
with open(os.path.join(path,jsonfile_name),'r',encoding='utf-8') as f:
	data=json.load(f)
for i in range(len(data)):
	if len(data[i]['ann']['labels'])==0:
		continue
	select=np.array(data[i]['ann']['labels'])!=3
	labels=[]
	bboxes=[]
	for j in range(len(data[i]['ann']['labels'])):
		if select[j]:
			labels.append(data[i]['ann']['labels'][j])
			bboxes.append(data[i]['ann']['bboxes'][j])

	bboxes=np.array(bboxes).tolist()
	data[i]['ann']['labels']=labels
	data[i]['ann']['bboxes']=bboxes


mmcv.dump(data,os.path.join(path,'bdd_all_2class.json'))
