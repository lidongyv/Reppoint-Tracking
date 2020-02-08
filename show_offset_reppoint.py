# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-12-11 16:41:28  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-12-11 16:41:28

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mmcv
import matplotlib
import os
import shutil
def kernel_inv_map(vis_attr, target_point, map_h, map_w):
	pos_shift = [vis_attr['dilation'] * 0 - vis_attr['pad'],
				 vis_attr['dilation'] * 1 - vis_attr['pad'],
				 vis_attr['dilation'] * 2 - vis_attr['pad']]
	source_point = []
	for idx in range(vis_attr['filter_size']**2):
		cur_source_point = np.array([target_point[0] + pos_shift[idx //3],
									 target_point[1] + pos_shift[idx % 3]])
		if cur_source_point[0] < 0 or cur_source_point[1] < 0 \
				or cur_source_point[0] > map_h - 1 or cur_source_point[1] > map_w - 1:
			continue
		source_point.append(cur_source_point.astype('f'))
	return source_point

def offset_inv_map(source_points, offset):
	# print(source_points,offset)
	for idx, _ in enumerate(source_points):
		source_points[idx][0] += offset[2*idx]
		source_points[idx][1] += offset[2*idx+1]
	return source_points

def get_bottom_position(vis_attr, top_points, all_offset):
	map_h = all_offset[0].shape[2]
	map_w = all_offset[0].shape[3]

	for level in range(vis_attr['plot_level']):
		source_points = []
		for idx, cur_top_point in enumerate(top_points):
			cur_top_point = np.round(cur_top_point)
			if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
				or cur_top_point[0] > map_h-1 or cur_top_point[1] > map_w-1:
				continue
			cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
			cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
			# print(cur_offset)
			cur_source_point = offset_inv_map(cur_source_point, cur_offset)
			source_points = source_points + cur_source_point
		top_points = source_points
	return source_points

def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, color=[255,0,0]):
	plot_area = vis_attr['plot_area']
	for idx, cur_source_point in enumerate(source_points):
		y = np.round((cur_source_point[0] + 0.5) * im.shape[0] / map_h).astype('i')
		x = np.round((cur_source_point[1] + 0.5) * im.shape[1] / map_w).astype('i')

		if x < 0 or y < 0 or x > im.shape[1]-1 or y > im.shape[0]-1:
			continue
		y = min(y, im.shape[0] - vis_attr['plot_area'] - 1)
		x = min(x, im.shape[1] - vis_attr['plot_area'] - 1)
		y = max(y, vis_attr['plot_area'])
		x = max(x, vis_attr['plot_area'])
		im[y-plot_area:y+plot_area+1, x-plot_area:x+plot_area+1, :] = np.tile(
			np.reshape(color, (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1)
		)
	return im


def show_dconv_offset_by_loc(im, all_offset, path,loc,step=[2, 2], filter_size=3,
					  dilation=1, pad=1, plot_area=1, plot_level=1,stride=8,refer=None):
	vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
				'plot_area': plot_area, 'plot_level': plot_level,'stride':stride}
	# print(all_offset[0])
	map_h = all_offset[0].shape[2]
	map_w = all_offset[0].shape[3]
	scale=1
	fig=plt.figure(figsize=(10, 3))
	count=0
	for (im_w,im_h) in loc:
		source_y = im_h
		source_x = im_w
		im_w=im_w/(im.shape[1] / map_w)
		im_h=im_h/(im.shape[0] / map_h)
		target_point = np.array([im_h,im_w]).astype(np.int)
		if source_y < plot_area or source_x < plot_area \
				or source_y >= im.shape[0] - plot_area or source_x >= im.shape[1] - plot_area:
			print('out of image')
			continue
		cur_im = np.copy(im)
		source_points = get_bottom_position(vis_attr, [target_point], all_offset)
		# target_offset=np.stack(source_points)
		# mean_offset=np.mean(target_offset,axis=0)
		# target_x=(mean_offset[1]* im.shape[1] / map_w).astype(np.int)
		# target_y=(mean_offset[0]* im.shape[0] / map_h).astype(np.int)

		cur_im = plot_according_to_point(vis_attr, cur_im, source_points, map_h, map_w)
		cur_im[source_y-4:source_y+4+1, source_x-4:source_x+4+1, :] = \
			np.tile(np.reshape([255, 255, 0], (1, 1, 3)), (2*4+1, 2*4+1, 1))
		# cur_im[target_y-4:target_y+4+1, target_x-4:target_x+4+1, :] = \
						# np.tile(np.reshape([0, 0, 255], (1, 1, 3)), (2*4+1, 2*4+1, 1))
		# valid_num=np.sum(np.where(cur_im[:,:,0]==255,1,0)*np.where(cur_im[:,:,1]==0,1,0)*np.where(cur_im[:,:,2]==0,1,0))
		# plt.text(0,0,'valid_offset:%d,offset_ratio:%.2f'%(valid_num,valid_num/6561),fontdict={'size': 10, 'color':  'black'})
		print('showing',im_h,im_w)
		plt.axis("off")
		plt.imshow(cur_im)
		fig.savefig(os.path.join(path,'%06d.jpg'%(count)),dpi=150)
		print(count)
		plt.clf()
		count+=1
	plt.close('all')


if __name__=='__main__':
	# torch.Size([2, 256, 48, 156])
	# torch.Size([2, 256, 24, 78])
	# torch.Size([2, 256, 12, 39])
	# torch.Size([2, 256, 6, 20])
	# torch.Size([2, 256, 3, 10])
	reppoints=mmcv.load('/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9_thres0.1_nms0.5_with2/refer/reppoints.pkl')
	path='/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9_thres0.1_nms0.5_with2/refer'
	classes = ['Car','Person','Cyclist']
	split=['not_detected','wrong_detected']
	for i in range(len(classes)):
		class_path=os.path.join(path,classes[i])
		video_name=os.listdir(class_path)
		for j in range(len(video_name)):
			video_path=os.path.join(class_path,video_name)
			
	loc,index, name=mmcv.load('/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9_thres0.1_nms0.5_with2/refer/Car/0001/wrong_detected/000006.pkl')

	print(len(reppoints))
	print(len(reppoints[0][index]))
	print(len(reppoints[0][index][0]))

	reppoints=reppoints[0][index][0][:1]
	img=matplotlib.image.imread(name)
	img=cv2.resize(img,(1242,374))
	

	loc=[loc.astype(np.int)[0][:2]]
	print(loc)
	scale=[8,16,32,64,128]


	save_path='/home/ld/RepPoints/ld_result/stsn_class_learn/epoch_9_thres0.1_nms0.5_with2/refer/Car/0001/wrong_detected/offset'
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	show_dconv_offset_by_loc(img,[reppoints],save_path,loc,plot_level=1,plot_area=3)
