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
import glob
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

def plot_according_to_point(vis_attr, im, source_points, map_h, map_w, color=[0,0,255]):
	plot_area = vis_attr['plot_area']
	for idx, cur_source_point in enumerate(source_points):
		y = np.round((cur_source_point[0] + 0.5) * 8).astype('i')
		x = np.round((cur_source_point[1] + 0.5) * 8).astype('i')
		# y = np.round((cur_source_point[0] + 0.5) * im.shape[0] / map_h).astype('i')
		# x = np.round((cur_source_point[1] + 0.5) * im.shape[1] / map_w).astype('i')

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


def show_dconv_offset_by_loc(im, all_offset, path,loc,loc_offset_scale,bias=None,step=[2, 2], filter_size=3,
					  dilation=1, pad=1, plot_area=1, plot_level=1,stride=8,agg=None):
	vis_attr = {'filter_size': filter_size, 'dilation': dilation, 'pad': pad,
				'plot_area': plot_area, 'plot_level': plot_level,'stride':stride}
	# print(all_offset[0])
	map_h = all_offset[0].shape[2]
	map_w = all_offset[0].shape[3]
	scale=1
	# fig=plt.figure(figsize=(10, 3))
	count=0
	for (im_w,im_h) in loc:


		source_y = im_h
		source_x = im_w
		# im_w=im_w/(im.shape[1] / map_w)
		# im_h=im_h/(im.shape[0] / map_h)

		im_w = (im_w / loc_offset_scale) / 8
		im_h = (im_h / loc_offset_scale) / 8
		if bias is not None:
			bias_h=np.clip(im_h.astype(np.int), a_min = 0, a_max = bias.shape[-1])
			bias_w=np.clip(im_w.astype(np.int), a_min = 0, a_max = bias.shape[-2])

			im_w=im_w+bias[0,1,bias_h,bias_w]
			im_h=im_h+bias[0,0,bias_h,bias_w]
			# im_wt=(im_w*(im.shape[1] / map_w)).astype(np.int)
			# im_ht=(im_h*(im.shape[0] / map_h)).astype(np.int)
			im_wt=(im_w*8*loc_offset_scale).astype(np.int)
			im_ht=(im_h*8*loc_offset_scale).astype(np.int)
		target_point = np.array([im_h,im_w]).astype(np.int)
		cur_im = np.copy(im)
		source_points = get_bottom_position(vis_attr, [target_point], all_offset)
		cur_im = plot_according_to_point(vis_attr, cur_im, source_points, map_h, map_w)
		if source_y < plot_area or source_x < plot_area \
				or source_y >= im.shape[0] - plot_area or source_x >= im.shape[1] - plot_area:
			print('out of image')
			continue
		if bias is not None:
			if im_ht < plot_area or im_wt < plot_area \
					or im_ht >= im.shape[0] - plot_area or im_wt >= im.shape[1] - plot_area:
				print('out of image')
				continue

		cur_im[source_y-plot_area:source_y+plot_area+1, source_x-plot_area:source_x+plot_area+1, :] = \
			np.tile(np.reshape([0, 255, 255], (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1))
		if bias is not None:
			cur_im[im_ht-plot_area:im_ht+plot_area+1, im_wt-plot_area:im_wt+plot_area+1, :] = \
				np.tile(np.reshape([0, 255, 0], (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1))
		im=np.copy(cur_im)

	print('showing',im_h,im_w)
	print(path)

	cv2.imwrite(path, cur_im)
	# plt.axis("off")
	# plt.imshow(cur_im)
	# fig.savefig(path,dpi=150)
	# plt.clf()
	# plt.close('all')


if __name__=='__main__':
	# torch.Size([2, 256, 48, 156])
	# torch.Size([2, 256, 24, 78])
	# torch.Size([2, 256, 12, 39])
	# torch.Size([2, 256, 6, 20])
	# torch.Size([2, 256, 3, 10])
	reppoints=mmcv.load('/home/zxn/RepPoints/zxn_result/debug/repoint_baseline_do3/kittibddwayo_epoch20_thres03_nms05_baseline_rerun_02142013_sub13/baseline/reppoints_13_all.pkl')
	offsets=mmcv.load('/home/zxn/RepPoints/zxn_result/debug/stsn_one_do3/kittibddwayo_epoch30_thres03_nms05_baseline/baseline/offset_13_all.pkl')
	path='/home/zxn/RepPoints/zxn_result/debug/stsn_one_do3/kittibddwayo_epoch30_thres03_nms05_baseline/baseline'
	classes = ['Car','Person','Cyclist']
	split=['addition','not_detected']
	scale = {'KITTI': 0.97, 'bdd': 1, 'waymo': 1.78}
	for i in range(len(classes)):
		class_path=os.path.join(path,classes[i])
		video_name=os.listdir(class_path)
		video_name.sort()
		for j in range(len(video_name)):
			video_path=os.path.join(class_path,video_name[j])

			for m in range(len(split)):
				video_split=os.path.join(video_path,split[m])
				pkls=glob.glob(video_split+'/*.pkl')
				pkls.sort()
				# 最内层的循环是这个，所有的addition
				for n in range(len(pkls)):
					print('154 debug')
					from IPython import embed
					embed()
					pkl_path=os.path.join(video_split,pkls[n])  #读取 addition的pkl文件
					# loc 应该是Bbox的中心点的位置，但是为啥是长度是4
					# 这个pkl 记录的信息是该文件中每个文件的信息
					# support1_name,support2_name：也是第一个detection文件生成的几个文件中图像的名称
					loc,index,img_name,support1_name,support2_name=mmcv.load(pkl_path)
					# kitti bdd waymo图像缩放的比例
					loc_offset_scale_sets = [0.97, 1, 1.7777]
					# target size = 736
					if img_name.split('/')[-3].split('_')[0] == 'kitti':
						loc_offset_scale = loc_offset_scale_sets[0]

					if img_name.split('/')[-3].split('_')[0] == 'bdd':
						loc_offset_scale = loc_offset_scale_sets[1]

					if img_name.split('/')[-3].split('_')[0] == 'waymo':
						loc_offset_scale = loc_offset_scale_sets[2]
					# 获取reppoints
					reppoint_frame=reppoints[0][index][0][:1,...]
					reppoint_support1 = reppoints[0][index-1][0][:1,...]
					reppoint_support2 = reppoints[0][index-5][0][:1,...]
					# print('175 debug')
					# from IPython import embed
					# embed()
					#
					# reppoint_support1=reppoints[0][index][0][1:2,...]
					# reppoint_support2=reppoints[0][index][0][2:3,...]
					#
					# print('177 debug')
					# from IPython import embed
					# embed()
					# reppoint_frame=reppoints[0][index][:1,...]
					# reppoint_support1=reppoints[0][index][1:2,...]
					# reppoint_support2=reppoints[0][index][2:3,...]

					offset_support1=offsets[0][0][index]  #(1, 2, 92, 160)
					offset_support2=offsets[1][0][index]  #(1, 2, 92, 160)
					# print(offset_support1.shape,offsets[1][0][index].shape)
					# exit()
					# print(offset_support1.shape,offset_support2.shape)
					# img=matplotlib.image.imread(img_name)

					# img=cv2.resize(img,(1242,374))
					# support1=matplotlib.image.imread(support1_name)
					# support1=cv2.resize(support1,(1242,374))
					# support2=matplotlib.image.imread(support2_name)
					# support2=cv2.resize(support2,(1242,374))


					# 读取这几张图像的作用是什么？
					img = cv2.imread(img_name)
					support1 = cv2.imread(support1_name)
					support2 = cv2.imread(support2_name)


					loc=[np.array(p).astype(np.int)[:2] for p in loc]  #只截取了前面的两个数，并且设置为int
					#current frame reppoints
					#用于构建一个新的文件夹
					img_name=img_name.split('/')
					offset_path='/'.join(img_name[:-1])+'/offset'
					if not os.path.exists(offset_path):
						os.mkdir(offset_path)
					offset_path=os.path.join(offset_path,img_name[-1])
					print(offset_path)
					#根据图像，reppoints，offset,loc 将offset绘制出来
					show_dconv_offset_by_loc(img,[reppoint_frame],offset_path,loc,loc_offset_scale, plot_level=1,plot_area=3)

					print('223 debug')
					from IPython import embed
					embed()
					#support1 offset
					support1_name=support1_name.split('/')
					offset_path='/'.join(support1_name[:-1])+'/offset'
					if not os.path.exists(offset_path):
						os.mkdir(offset_path)
					offset_path=os.path.join(offset_path,support1_name[-1])
					print(offset_path)
					show_dconv_offset_by_loc(support1,[reppoint_support1],offset_path,loc,loc_offset_scale,bias=offset_support1,plot_level=1,plot_area=3)

					print('235 debug')
					from IPython import embed
					embed()
					#support2 offset
					support2_name=support2_name.split('/')
					offset_path='/'.join(support2_name[:-1])+'/offset'
					if not os.path.exists(offset_path):
						os.mkdir(offset_path)
					offset_path=os.path.join(offset_path,support2_name[-1])
					print(offset_path)
					show_dconv_offset_by_loc(support2,[reppoint_support2],offset_path,loc,loc_offset_scale,bias=offset_support2,plot_level=1,plot_area=3)
				# exit()
