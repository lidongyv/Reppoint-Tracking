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
	for idx, _ in enumerate(source_points):
		source_points[idx][0] += offset[2*idx]
		source_points[idx][1] += offset[2*idx+1]
	return source_points

def get_bottom_position_mask(vis_attr, top_points, all_offset,mask=None):
	map_h = all_offset[0].shape[2]
	map_w = all_offset[0].shape[3]
	
	for level in range(vis_attr['plot_level']):
		source_points = []
		source_weight=[]
		for idx, cur_top_point in enumerate(top_points):
			cur_top_point = np.round(top_points[idx])
			if level==0:
				mask_weight_init=1
			else:
				mask_weight_init=top_weight[idx]
			if cur_top_point[0] < 0 or cur_top_point[1] < 0 \
				or cur_top_point[0] > map_h-1 or cur_top_point[1] > map_w-1:
				continue
			cur_source_point = kernel_inv_map(vis_attr, cur_top_point, map_h, map_w)
			cur_offset = np.squeeze(all_offset[level][:, :, int(cur_top_point[0]), int(cur_top_point[1])])
			cur_source_point = offset_inv_map(cur_source_point, cur_offset)
			source_points = source_points + cur_source_point

			source_weight=source_weight+(np.squeeze(mask[level] \
				[:, :, int(cur_top_point[0]), int(cur_top_point[1])])*mask_weight_init).tolist()
		top_points = source_points
		top_weight = source_weight
	return source_points,source_weight

def plot_according_to_point_mask(vis_attr, im, source_points, map_h, map_w, color=[1.0,0.0,0.0],mask=None):
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

		t_color=np.array(color.copy())
		if mask[idx]<0.3:
			continue
		if mask[idx]>0.5:
			t_color=[mask[idx],0.0,0.0]
		else:
			t_color=[0.0,0.0,1-mask[idx]]
		print(t_color,np.max(img))
		im[y-plot_area:y+plot_area+1, x-plot_area:x+plot_area+1, :] = np.tile(
			np.reshape(t_color, (1, 1, 3)), (2*plot_area+1, 2*plot_area+1, 1)
		)
	return im


def show_mask_dcn_by_loc(im, all_offset, path,loc,step=[2, 2], filter_size=3,
					  dilation=1, pad=1, plot_area=1, plot_level=1,stride=8,refer=None,mask=None):
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
		source_points,source_mask = get_bottom_position_mask(vis_attr, [target_point], all_offset,mask=mask)
		source_mask=source_mask/np.max(source_mask)
		cur_im = plot_according_to_point_mask(vis_attr, cur_im, source_points, map_h, map_w,mask=source_mask)
		cur_im[source_y-4:source_y+4+1, source_x-4:source_x+4+1, :] = \
			np.tile(np.reshape([1.0, 1.0, 0.0], (1, 1, 3)), (2*4+1, 2*4+1, 1))
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
	for i in range(6):
		# i=5
		path='/home/ld/RepPoints/offset/agg_st_stsn_dcnv2_mean_kernel_fix_rep_support_class3_trans/epoch14_fix_train/'+str(i)
		print('video:',i)
		# path='/home/ld/RepPoints/offset_back/agg_st/2'
		files=os.listdir(path)
		files.sort()
		img_files=[]
		for p in files:
			if p.endswith('.png'):
				img_files.append(p)
		img=matplotlib.image.imread(os.path.join(path,img_files[0]))
		imgs=matplotlib.image.imread(os.path.join(path,img_files[1]))
		img=cv2.resize(img,(1242,374))
		imgs=cv2.resize(imgs,(1242,374))

		# save_path=os.path.join(path,'support_f')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# else:
		#	 shutil.rmtree(save_path)
		#	 os.mkdir(save_path)
		# support_f=np.load(path+'/support_f.npy')
		# show_feature(support_f,save_path)
		# save_path=os.path.join(path,'agg_f')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# else:
		#	 shutil.rmtree(save_path)
		#	 os.mkdir(save_path)
		# agg_f=np.load(path+'/agg_f.npy')
		# show_feature(agg_f,save_path)
		
		# save_path=os.path.join(path,'refer_f')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# agg_f=np.load(path+'/refer_f.npy')
		# show_feature(agg_f,save_path)

		# offset0=np.load(path+'/resnetl20.npy')
		# offset1=np.load(path+'/resnetl21.npy')
		# offset2=np.load(path+'/resnetl22.npy')
		# offset3=np.load(path+'/resnetl23.npy')
		loc=np.load(path+'/box_loc.npy')
		print(loc.shape)
		if loc.shape[1]==3:
			continue
		loc=loc[np.where(loc[:,3]>0.3)]
		
		loc=loc.astype(np.int)
		scale=[8,16,32,64,128]
		for m in range(5):
			print('stage_',m)
			offset=[]
			mask=[]
			for n in range(4):
				try:
					offset_mask=np.load(path+'/stage_'+str(m)+'_offset_'+str(n)+'.npy')
				except:
					break
				offset.append(offset_mask[:,:18,:,:])
				mask.append(offset_mask[:,-9:,:,:])
			# offset=[np.load(path+'/stage_'+str(m)+'.npy')]
			# mask=[np.load(path+'/mask_'+str(m)+'.npy')]

			save_path=os.path.join(path,'stage_'+str(m))
			if not os.path.exists(save_path):
				os.mkdir(save_path)
			else:
				shutil.rmtree(save_path)
				os.mkdir(save_path)

			loc_select=loc[np.where(loc[:,2]==scale[m]),:2]
			# loc_select=loc[np.where(loc[:,2]>0),:2]
			show_mask_dcn_by_loc(imgs,offset,save_path,loc_select[0],plot_level=len(offset),plot_area=3,mask=mask)
			# exit()
		# exit()
		# offset1=np.load(path+'/resnetl20.npy')
		# offset2=np.load(path+'/resnetl21.npy')
		# offset3=np.load(path+'/resnetl22.npy')
		# offset4=np.load(path+'/resnetl23.npy')
		# print(offset1.shape)
		# save_path=os.path.join(path,'resnet')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(img,[offset4,offset3,offset2,offset1],save_path)
		# offset1=np.load(path+'/support_00.npy')
		# offset2=np.load(path+'/support_01.npy')
		# offset3=np.load(path+'/support_02.npy')
		# offset4=np.load(path+'/support_03.npy')
		# print(offset1.shape)
		# offset4=np.load(path+'/offset4.npy')
		# offset4=offset4
		# print(offset4.shape)
		# save_path=os.path.join(path,'offset4')
		# # offset4=np.load('/home/ld/RepPoints/offset_back/agg_st/2/init_rep.npy')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(imgs,[offset4],save_path,step=[1, 1],plot_level=1,plot_area=3)
		# save_path=os.path.join(path,'offset123')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(imgs,[offset3,offset2,offset1],save_path,step=[2, 2],refer=img,plot_level=3)

		# offset1=np.load(path+'/stsn_r1.npy')
		# offset2=np.load(path+'/stsn_r2.npy')
		# offset3=np.load(path+'/stsn_r2.npy')
		# offset4=np.load(path+'/stsn_r4.npy')
		# print(offset1.shape)
		# save_path=os.path.join(path,'stsn_r')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(img,[offset4,offset3,offset2,offset1],save_path)
		

		# offset1=np.load(path+'/init_rep.npy')
		# print(offset1.shape)
		# save_path=os.path.join(path,'init_rep')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(img,[offset1],save_path,plot_level=1,plot_area=3)
		# offset2=np.load(path+'/refine_rep.npy')
		# print(offset2.shape)
		# save_path=os.path.join(path,'refine_rep')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(img,[offset2,offset1],save_path,plot_level=2,plot_area=3)

		# offset0=np.load(path+'/init_rep.npy')
		# offset1=np.load(path+'/stsn_r1.npy')
		# offset2=np.load(path+'/stsn_r2.npy')
		# offset3=np.load(path+'/stsn_r2.npy')
		# offset4=np.load(path+'/stsn_r4.npy')
		# print(offset1.shape)
		# save_path=os.path.join(path,'repstn_r')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(img,[offset3,offset2,offset1,offset0],save_path)
		
		# offset1=np.load(path+'/stsn_s1.npy')
		# offset2=np.load(path+'/stsn_s2.npy')
		# offset3=np.load(path+'/stsn_s3.npy')
		# offset4=np.load(path+'/stsn_s4.npy')
		# print(offset1.shape)
		# save_path=os.path.join(path,'repstn_s')
		# if not os.path.exists(save_path):
		#	 os.mkdir(save_path)
		# show_dconv_offset(imgs,[offset3,offset2,offset1,offset0],save_path)
		# weight=np.load(os.path.join(path,'weight.npy'))
		# print(weight.shape)
		# draw_weight(img,imgs,weight,path)