# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-25 19:24:06  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-25 19:24:06

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import time
import ffmpeg
import json
import numpy as np
import cv2
# config_file = '/home/ld/RepPoints/configs/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py'
# checkpoint_file = '/home/ld/RepPoints/trained/faster_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-e41688c9.pth'
# config_file = '/home/ld/RepPoints/mmdetection/configs/reppoints/reppoints_moment_r101_dcn_fpn_2x.py'
# checkpoint_file = '/home/ld/RepPoints/trained/reppoints_moment_r101_dcn_fpn_2x.pth'
config_file ='/home/ld/RepPoints/configs/reppoints_moment_r101_dcn_fpn_kitti_agg_fuse_st.py'

checkpoint_file='/home/ld/RepPoints/final/fuse/epoch_19.pth'
# config_file ='/home/ld/RepPoints/configs/retinanet_r101_fpn_kitti.py'
# checkpoint_file='/home/ld/RepPoints/work_dirs/retinanet_r101_fpn_kitti/epoch_50.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
data_path='/backdata01/KITTI/kitti/tracking'
# ['calib', 'disparity', 'image_02', 'label_02', 'splits', 'velodyne', 'velodyne_img', 
# 'Flow', 'Inv_Flow', 'trackinglabel_02', 'kitti.json', 'shuffle data', 'kitti_train.json'
#  'kitti_val.json']
jsonfile_name='kitti_val_3class.json'
# test a video and show the results
# with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
# 	data=json.load(f)
compute_time=0
eval_data=[]
out_path='/home/ld/RepPoints/offset/reppoints_moment_r101_dcn_fpn_kitti_agg_fuse_st_test'
if not os.path.exists(out_path):
	os.mkdir(out_path)
data=[]
eval_data_path='/backdata01/KITTI/kitti/tracking/training/eval_img/'
scene_path=[]
for i in range(len(os.listdir(eval_data_path))):
	scene=[]
	scene_path.append(os.path.join(out_path,str(i)))
	if not os.path.exists(scene_path[i]):
		os.mkdir(scene_path[i])
	for j in os.listdir(os.path.join(eval_data_path,str(i))):
		scene.append(os.path.join(eval_data_path,str(i),j))
	data.append(scene)
# print(data)
# data=[data[2]]
# scene_path=[scene_path[2]]
for i,(scene) in enumerate(data):
	# img = mmcv.imread(os.path.join(data_path,img_name))
	img=scene[0]
	img1=scene[2]
	print(i,scene)
	result,box_loc = inference_detector(model, [img,img1])
	# result = inference_detector(model, [img,img1])
	# np.save(scene_path[i]+'/offset4'+'.npy',model.agg.offset4.data.cpu().numpy())
	# np.save(scene_path[i]+'/support_f'+'.npy',model.agg.support_f.data.cpu().numpy())
	# np.save(scene_path[i]+'/agg_f'+'.npy',model.agg.agg_f.data.cpu().numpy())
	# np.save(scene_path[i]+'/refer_f'+'.npy',model.agg.refer_f.data.cpu().numpy())
	show_result(img, result, model.CLASSES, score_thr=0.15,show=False,out_file=os.path.join(scene_path[i],'agg'+img.split('/')[-1]))
	show_result(img1, result, model.CLASSES, score_thr=0.15,show=False,out_file=os.path.join(scene_path[i],'agg'+img1.split('/')[-1]))
	print('offset lenght',len(model.agg.offset))
	# for m in range(len(model.agg.offset)):
	# 	for n in range(len(model.agg.offset[m])):
	# 		np.save(scene_path[i]+'/stage_'+str(m)+'_offset_'+str(n)+'.npy',model.agg.offset[m][n].data.cpu().numpy())
	# 		print(model.agg.offset[m][n].shape)
	for m in range(len(model.agg.offset)):
		np.save(scene_path[i]+'/offset_'+str(m)+'.npy',model.agg.offset[m].data.cpu().numpy())
	for m in range(len(model.agg.mask)):
		np.save(scene_path[i]+'/mask_'+str(m)+'.npy',model.agg.mask[m].data.cpu().numpy())
	# 	print(model.agg.offset[m].shape)
	print(box_loc.shape)
	np.save(scene_path[i]+'/box_loc'+'.npy',box_loc.data.cpu().numpy())
	continue
	# np.save(scene_path[i]+'/offset4'+'.npy',model.agg.offset4.data.cpu().numpy())
	# continue
	# exit()
	# np.save(scene_path[i]+'/offset4.npy',model.agg.conv4.weight.data.cpu().numpy())
	# np.save(scene_path[i]+'/offset4'+'.npy',model.agg.offset4.data.cpu().numpy())
	# np.save(scene_path[i]+'/support_f'+'.npy',model.agg.support_f.data.cpu().numpy())
	# np.save(scene_path[i]+'/agg_f'+'.npy',model.agg.agg_f.data.cpu().numpy())
	# np.save(scene_path[i]+'/refer_f'+'.npy',model.agg.refer_f.data.cpu().numpy())
	# np.save(scene_path[i]+'/box_loc'+'.npy',box_loc.data.cpu().numpy())
	
	# for j in range(len(model.backbone.layer2)):
	# 	print(model.backbone.layer2[j])
	# 	print(model.backbone.layer2[j].offset.shape)
	# 	# print(model.backbone.layer2[i].offset)
	# 	np.save(scene_path[i]+'/resnetl2'+str(j)+'.npy',model.backbone.layer2[j].offset[1:2,:,:,:].data.cpu().numpy())
	# continue
	# show_result(img, result, model.CLASSES, show=False,out_file=os.path.join(scene_path[i],'s64'+img.split('/')[-1]))
	# show_result(img1, result, model.CLASSES, show=False,out_file=os.path.join(scene_path[i],'s64'+img1.split('/')[-1]))
	# continue
	# if not os.path.exists(os.path.join(out_path,video_name)):
	# 	os.mkdir(os.path.join(out_path,video_name))
	# show_result(img, result, model.CLASSES, show=False,out_file=os.path.join('/home/ld/RepPoints/offset/',img_name.split('/')[-1]))
	# exit()
	#offset
	# print(model.CLASSES)
	# for j in range(len(model.backbone.layer2)):
	# 	print(model.backbone.layer2[j])
	# 	print(model.backbone.layer2[j].offset.shape)
	# 	# print(model.backbone.layer2[i].offset)
	# 	np.save('/home/ld/RepPoints/offset/agg_st/resnetl2'+str(j)+'.npy',model.backbone.layer2[j].offset[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/resnetl2'+str(j)+'.npy',model.backbone.layer2[j].offset[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_r1'+'.npy',model.agg.roffset1[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_r2'+'.npy',model.agg.roffset2[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_r3'+'.npy',model.agg.roffset3[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_r4'+'.npy',model.agg.roffset4[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_s1'+'.npy',model.agg.soffset1[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_s2'+'.npy',model.agg.soffset2[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_s3'+'.npy',model.agg.soffset3[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/stsn_s4'+'.npy',model.agg.soffset4[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/init_rep'+'.npy',model.bbox_head.init_offset[:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/refine_rep'+'.npy',model.bbox_head.refine_offset[:1,:,:,:].data.cpu().numpy())
	for m in range(5):
		for n in range(len(model.agg.offset[m])):
			np.save(scene_path[i]+'/support_'+str(m)+str(n)+'.npy',model.agg.offset[m][n][:1,:,:,:].data.cpu().numpy())
	# np.save('/home/ld/RepPoints/offset/agg_st/weight'+'.npy',model.agg.weight.data.cpu().numpy())
	show_result(img, result, model.CLASSES, show=False,out_file=os.path.join(scene_path[i],img.split('/')[-1]))
	show_result(img1, result, model.CLASSES, show=False,out_file=os.path.join(scene_path[i],img1.split('/')[-1]))
	continue
	if isinstance(result, tuple):
		bbox_result, segm_result = result
	else:
		bbox_result, segm_result = result, None
	#four value and one score
	bboxes = np.vstack(bbox_result)
	scores = bboxes[:, -1]
	inds = scores > 0.5
	bboxes = bboxes[inds, :][:,:4]
	
	# draw bounding boxes
	labels = [
		np.full(bbox.shape[0], i, dtype=np.int32)
		for i, bbox in enumerate(bbox_result)
	]
	labels = np.concatenate(labels)
	labels = labels[inds]
	frame_data={"video_id":frame['video_id'],"filename":os.path.join(frame['filename']), \
		"ann":{"bboxes":bboxes.tolist(),"labels":labels.tolist(), \
			"track_id":labels.tolist()},"flow_name":frame['flow_name'],"inv_flow_name":frame['inv_flow_name']}
	eval_data.append(frame_data)
	
with open(os.path.join('./result/','retina_'+jsonfile_name),'w+',encoding='utf-8') as f:
	data=json.dump(eval_data,f)
