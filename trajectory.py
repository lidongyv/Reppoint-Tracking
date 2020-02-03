# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-12-06 00:02:47  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-12-06 00:02:47
import os
import numpy as torch
import torch
import cv2
from matplotlib import pyplot as plt
from IO import *
import json
import mmcv
kitti_height=370
kitti_width=1224

class Trajectory():
	def __init__(self):
		self.length=0
		self.max_id=0
		self.template=torch.ones([2000,2000,2]).float()
		self.template[:,:,0]*=torch.reshape(torch.arange(2000).float(),(2000,1))
		self.template[:,:,1]*=torch.reshape(torch.arange(2000).float(),(1,2000))
		self.reset=True
		self.one=torch.ones(1).float()
		self.zero=torch.zeros(1).float()
		self.tr=[]
		self.tr_id=[]
		self.slice=[]
		self.id_aligned=[]
		self.id_len=[]

	def update(self,fore,back):
		template=self.template[:back.shape[0],:back.shape[1],:]
		if self.reset:
			id=torch.reshape(torch.arange(1,template.shape[0]*template.shape[1]+1), \
				(template.shape[0],template.shape[1]))
			self.slice.append(id)
			self.slice_volume=torch.stack(self.slice)
			self.max_id=torch.max(id)
			self.id_aligned=torch.cat([self.one.long(),id.view(-1)])
			self.id_len=torch.ones_like(self.id_aligned)
			self.reset=False
		else:
			flow_check,fore_warp=self.check(fore,back)
			id=torch.zeros_like(self.slice[-1])
			id[[fore_warp[:,:,0],fore_warp[:,:,1]]]=self.slice[-1]*flow_check.long()
			# self.id_len[id]+=1
			new=torch.where(id>0,self.zero,self.one).nonzero(as_tuple=True)
			id[new]=torch.arange(len(new[0]))+self.max_id+1
			# self.cut(flow_check)
			# self.id_aligned=torch.cat([self.id_aligned,id[new]])
			# self.id_len=torch.cat([self.id_len,torch.ones_like(id[new])])
			self.max_id=torch.max(id)
			self.slice.append(id)
			# self.slice_volume=torch.stack(self.slice)
	def get_tr(self,id):
		trajectory=torch.where(self.slice_volume==id,self.one,self.zero).nonzero(as_tuple=True)
		return trajectory
	def cut(self,flow_check):
		select=(1-flow_check).nonzero(as_tuple=True)
		id_select=self.slice[-1][select]
		len_check=torch.where(self.id_len[id_select]>10,self.one,self.zero)
		id_checked=id_select[len_check.nonzero(as_tuple=True)]
		if len(id_checked)==0:
			return 0
		self.tr_id.append(id_checked)
		print(len(id_checked))
		for i in id_checked:
			self.tr.append(self.get_tr(i))

	def check(self,fore,back):
		id=self.slice[0]
		template=self.template[:fore.shape[0],:fore.shape[1],:]
		fore_warp=torch.round(template+fore).long()
		fore_warp[:,:,0]=torch.clamp(fore_warp[:,:,0],min=0,max=fore.shape[0]-1)
		fore_warp[:,:,1]=torch.clamp(fore_warp[:,:,1],min=0,max=fore.shape[1]-1)
		fore_result=torch.where(torch.norm(back[fore_warp[:,:,0],fore_warp[:,:,1]]+fore,2,dim=2)<=1, \
			self.one,self.zero)
		back_warp=torch.round(template+back).long()
		back_warp[:,:,0]=torch.clamp(back_warp[:,:,0],min=0,max=fore.shape[0]-1)
		back_warp[:,:,1]=torch.clamp(back_warp[:,:,1],min=0,max=fore.shape[1]-1)
		back_result=torch.where(torch.norm(fore[back_warp[:,:,0],back_warp[:,:,1]]+back,2,dim=2)<=1, \
			self.one,self.zero)

		fore*=fore_result.unsqueeze(-1)
		back*=back_result.unsqueeze(-1)
		fore_warp=torch.round(template+fore).long()
		fore_warp[:,:,0]=torch.clamp(fore_warp[:,:,0],min=0,max=fore.shape[0]-1)
		fore_warp[:,:,1]=torch.clamp(fore_warp[:,:,1],min=0,max=fore.shape[1]-1)
		fore_result=torch.where(torch.norm(back[fore_warp[:,:,0],fore_warp[:,:,1]]+fore,2,dim=2)<=1, \
			self.one,self.zero)
		return fore_result,fore_warp
	def _reset(self):
		self.reset=True
		self.length=0
		self.max_id=0
		self.tr=[]
		self.tr_id=[]
		self.id_list=[]
		self.id_len=[]
		self.slice=[]
		self.id_aligned=[]
		self.id_pooled=[]
if __name__=='__main__':
	trajectory=Trajectory()
	data_path='/backdata01/KITTI/kitti/tracking'
	jsonfile_name='kitti_val.json'
	output_path='/backdata01/KITTI/kitti/trajectory'
	with open(os.path.join(data_path,jsonfile_name),'r',encoding='utf-8') as f:
		data=json.load(f)
	video_name=None
	video_length=0
	json_data_iou=[]
	json_data_flow=[]
	for i in range(len(data)):
		#print(i,'in',len(data))
		info=data[i]
		if info['video_id']!=video_name:
			video_length=1
			video_name=info['video_id']
			trajectory._reset()
		else:
			video_length+=1
		if video_name!='0020':
			continue
		#print('video_id',video_name,'image_name',info['filename'],'processing frame',video_length)
		image_name=os.path.join(data_path,info['filename'])
		image = mmcv.imread(image_name)
		flow_name=os.path.join(data_path,data[i-1]['flow_name'])
		inv_flow_name=os.path.join(data_path,data[i]['inv_flow_name'])
		print(video_length)
		flow=torch.from_numpy(readFlow(flow_name))
		inv_flow=torch.from_numpy(readFlow(inv_flow_name))
		# print(torch.max(flow),torch.min(flow))
		trajectory.update(flow,inv_flow)
		if i==len(data)-1 or data[i+1]['video_id']!=video_name:
			print('saving trajectory of', video_name)
			torch.save({'trajectory volume':torch.stack(trajectory.slice).long(),'max_id':trajectory.max_id}, \
				os.path.join(output_path,video_name+'.pkl'))

		#print(trajectory.max_id)
		# color=flow_to_color(flow,convert_to_bgr=True)
		# plt.imshow(color)
		# plt.show()
		# exit()
