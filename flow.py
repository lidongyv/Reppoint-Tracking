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

data_path='/backdata01/KITTI/kitti/tracking'
outpath=os.path.join('/backdata01/KITTI/kitti/tracking/training/inv_flow_vis','0017')
# os.mkdir(outpath)
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
	if info['video_id']!='0017':
		continue
	print('video_id',info['video_id'],'image_name',info['filename'],'processing frame',video_length)
	image_name=os.path.join(data_path,info['filename'])
	flow_name=os.path.join(data_path,data[i]['flow_name'])
	# image = mmcv.imread(image_name)
	flow=np.array(readFlow(flow_name))
	#print(flow)
	print(os.path.join(data_path,data[i-1]['flow_name']))
	exit()
	color=flow_to_color(flow,convert_to_bgr=True)
	print(os.path.join('/backdata01/KITTI/kitti/tracking/training/inv_flow_vis',info['video_id'],info['filename']))
	cv2.imwrite(os.path.join(outpath,info['filename'].split('/')[-1]),color)
	# plt.imshow(color)
	# plt.show()

