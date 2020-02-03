# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-12-11 16:41:28  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-12-11 16:41:28
from mmdet.apis import show_result
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import matplotlib
import os
import shutil
import json
if __name__=='__main__':
	# torch.Size([2, 256, 48, 156])
	# torch.Size([2, 256, 24, 78])
	# torch.Size([2, 256, 12, 39])
	# torch.Size([2, 256, 6, 20])
	# torch.Size([2, 256, 3, 10])

	data_path='/backdata01/KITTI/kitti/tracking'
	jsonfile_name='kitti_val_3class.json'
	# test a video and show the results
	with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
		data=json.load(f)
	out_path='/home/ld/RepPoints/debug/stsn_one_fuse_1_27_2/epoch14_thres0.1_nms0.5_with2_val_offset/frame_1'
	det_record=mmcv.load(os.path.join(out_path,'det_result.pkl'))
	loc_record=mmcv.load(os.path.join(out_path,'loc_result.pkl'))
	offset_record=mmcv.load(os.path.join(out_path,'offset.pkl'))
	classes = ['Vehicle','Pedestrian','Cyclist']
	video_name_check=None
	for i,(frame) in enumerate(data):
		print(i,'in',len(data))
		video_name=frame['video_id']
		if video_name_check is None:
			video_name_check=video_name
			video_length=0
			if not os.path.exists(os.path.join(out_path,video_name_check)):
				os.mkdir(os.path.join(out_path,video_name_check))
				os.mkdir(os.path.join(out_path,video_name_check,'all_offset'))
		else:
			if video_name_check==video_name:
				video_length+=1
			else:
				video_name_check=video_name
				video_length=0
				if not os.path.exists(os.path.join(out_path,video_name_check)):
					os.mkdir(os.path.join(out_path,video_name_check))
					os.mkdir(os.path.join(out_path,video_name_check,'all_offset'))
		print('video_name',video_name,'image_name',frame['filename'],'video_length',video_length)
		img_name=frame['filename']
		# img = mmcv.imread(os.path.join(data_path,img_name))
		img=os.path.join(data_path,img_name)
		img_list=[img]

		if data[i-2]['video_id']==video_name:
			img_list.append(os.path.join(data_path,data[i-2]['filename']))
		else:
			img_list.append(os.path.join(data_path,data[i]['filename']))
		# if i+2>=len(data):
		# 	img_list.append(os.path.join(data_path,data[i]['filename']))
		# else:
		# 	if data[i+2]['video_id']==video_name:
		# 		img_list.append(os.path.join(data_path,data[i+2]['filename']))
		# 	else:
		# 		img_list.append(os.path.join(data_path,data[i]['filename']))

		img=cv2.imread(img_list[0])
		img1=cv2.imread(img_list[1])
		img=cv2.resize(img,(1242,374))
		img1=cv2.resize(img1,(1242,374))
		show_img=img1
		print(img_list[0])
		loc=loc_record[i]
		if loc.shape[1]==3:
			continue
		offset=offset_record[i]
		scale=[8,16,32,64,128]
		det=det_record[i]
		ps=4
		if video_length%100==0:
			for m in range(len(loc)):
				v_im=show_img+0
				x=(loc[m,1]).long().clamp(min=ps+1,max=374-ps-1)
				y=(loc[m,0]).long().clamp(min=ps+1,max=1242-ps-1)
				v_im[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
					np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
				
				bias=offset[m]*loc[m,2]
				bias=np.floor(bias).long()
				x=(loc[m,1]+bias[0]).long().clamp(min=ps+1,max=374-ps-1)
				y=(loc[m,0]+bias[1]).long().clamp(min=ps+1,max=1242-ps-1)
				v_im[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
					np.tile(np.reshape([0, 0, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
				show_result(v_im, det, classes, show=False,out_file=os.path.join(out_path,video_name,'all_offset',str(m)+img_list[0].split('/')[-1]))
		for m in range(len(loc)):
			x=(loc[m,1]).long().clamp(min=ps+1,max=374-ps-1)
			y=(loc[m,0]).long().clamp(min=ps+1,max=1242-ps-1)
			show_img[x-ps:x+ps+1, y-ps:y+ps+1, :] = \
				np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))
		
			bias=offset[m]*loc[m,2]
			bias=np.floor(bias).long()
			x=(loc[m,1]+bias[0]).long().clamp(min=ps+1,max=374-ps-1)
			y=(loc[m,0]+bias[1]).long().clamp(min=ps+1,max=1242-ps-1)
			show_img[x-ps:x+ps+1, y-ps:y+ps+1,:] = \
				np.tile(np.reshape([0, 0, 255], (1, 1, 3)), (2*ps+1, 2*ps+1, 1))

		img=show_result(show_img, det, classes, show=False,out_file=os.path.join(out_path,video_name,img_list[0].split('/')[-1]),out=True)
		